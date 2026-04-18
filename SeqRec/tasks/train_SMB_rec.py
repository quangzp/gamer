import os
import csv
import copy
import json
import torch
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader, ConcatDataset

from SeqRec.tasks.base import Task
from SeqRec.datasets.SMB_dis_dataset import SMBDisDataset, SMBDisUserLevelDataset
from SeqRec.datasets.loading_SMB_dis import load_SMBDis_datasets, load_SMBDis_test_dataset
from SeqRec.datasets.collator_traditional import TraditionalCollator, TraditionalTestCollator, TraditionalUserLevelCollator
from SeqRec.modules.model_base.seq_model import SeqModel
from SeqRec.models.discriminative.GRU4Rec import GRU4Rec, GRU4RecConfig
from SeqRec.models.discriminative.SASRec import SASRec, SASRecConfig
from SeqRec.models.discriminative.BERT4Rec import BERT4Rec, BERT4RecConfig
from SeqRec.models.discriminative.MBHT import MBHT, MBHTConfig
from SeqRec.models.discriminative.MBSTR import MBSTR, MBSTRConfig
from SeqRec.models.discriminative.PBAT import PBAT, PBATConfig
from SeqRec.models.discriminative.END4Rec import END4Rec, END4RecConfig
from SeqRec.utils.config import Config
from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args
from SeqRec.utils.pipe import set_seed
from SeqRec.utils.pipe import get_tqdm


class TrainSMBRec(Task):
    """
    Train a SMB recommender for discriminative models.
    """

    @staticmethod
    def parser_name() -> str:
        return "train_SMB_rec"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TrainSMBRec task.
        """
        parser = sub_parsers.add_parser(
            "train_SMB_rec", help="Train a recommender for session-wise multi-behavior recommendation."
        )
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parser.add_argument(
            "--add_uid",
            action="store_true",
            help="Whether to add user id in the dataset.",
        )
        parser.add_argument(
            "--optim", type=str, default="adamw", help="The name of the optimizer"
        )
        parser.add_argument(
            "--epochs", type=int, default=200, help="Number of training epochs"
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=5e-4,
            help="Learning rate for the optimizer",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=256,
            help="Batch size during training",
        )
        parser.add_argument(
            "--logging_step", type=int, default=30, help="Logging frequency in steps"
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.01,
            help="Weight decay for regularization",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=20,
            help="Number of evaluation steps to wait before stopping training if no improvement",
        )
        parser.add_argument(
            "--test_task",
            type=str,
            default="smb_dis",
            help="test task",
        )
        parser.add_argument(
            "--metrics",
            type=str,
            default="hit@1,hit@5,hit@10,recall@1,recall@5,recall@10,ndcg@5,ndcg@10",
            help="test metrics, separate by comma",
        )
        parser.add_argument(
            "--wandb_run_name",
            type=str,
            default="default",
            help="Name for the Weights & Biases run",
        )
        parser.add_argument(
            "--result_dir",
            type=str,
            default="./results",
            help="The output directory",
        )
        parser.add_argument(
            '--only_test',
            action='store_true',
            help='Only perform testing without training.',
        )
        parser.add_argument(
            "--export_submission",
            action="store_true",
            help="Export challenge submission file after testing.",
        )
        parser.add_argument(
            "--submission_file",
            type=str,
            default="",
            help="Submission csv path. If empty, save to result_dir/submission-<test_task>.csv",
        )
        parser.add_argument(
            "--submission_topk",
            type=int,
            default=10,
            help="Top-K jobs per session in submission.",
        )
        parser.add_argument(
            "--uid_map_file",
            type=str,
            default="",
            help="Path to <dataset>.uid_to_original_session.json",
        )
        parser.add_argument(
            "--item_map_file",
            type=str,
            default="",
            help="Path to <dataset>.item_to_raw_job.json",
        )
        parser.add_argument(
            "--apply_behavior_label",
            type=str,
            default="apply",
            help="Behavior label treated as applies_for=True.",
        )

    def test_single_behavior(self, data_loader: DataLoader, behavior: str) -> dict:
        eval_results = {metric: [] for metric in self.metric_list}
        user_metric_dict: dict[str, dict[int, float]] = {m: {} for m in self.metric_list}
        with torch.no_grad():
            for batch, targets in get_tqdm(data_loader, desc=f"{behavior} testing"):
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                scores: torch.Tensor = self.model.full_sort_predict(batch)
                scores = scores.cpu().numpy()
                ranks = np.argsort(-scores, axis=1)
                for metric in self.metric_list:
                    for batch_i, (single_ranks, single_targets) in enumerate(zip(ranks, targets)):
                        single_targets = list(set(single_targets))
                        metric_name, k = metric.split('@')
                        k = int(k)
                        if metric_name == "hit":
                            hit = np.isin(single_targets, single_ranks[:k])
                            eval_results[metric].append(float(np.any(hit)))
                            if 'uid' in batch:
                                uid = batch['uid'][batch_i].item()
                                user_metric_dict[metric][uid] = float(np.any(hit))
                        elif metric_name == "recall":
                            recall = np.isin(single_targets, single_ranks[:k])
                            eval_results[metric].append(np.mean(recall.astype(float)))
                            if 'uid' in batch:
                                uid = batch['uid'][batch_i].item()
                                user_metric_dict[metric][uid] = np.mean(recall.astype(float))
                        elif metric_name == "ndcg":
                            dcg = 0.0
                            idcg = 0.0
                            for i in range(len(single_targets)):
                                rank = np.where(single_ranks == single_targets[i])[0][0]
                                if rank < k:
                                    dcg += 1.0 / np.log2(rank + 2)
                            for i in range(min(len(single_targets), k)):
                                idcg += 1.0 / np.log2(i + 2)
                            ndcg = dcg / idcg if idcg > 0 else 0.0
                            eval_results[metric].append(ndcg)
                            if 'uid' in batch:
                                uid = batch['uid'][batch_i].item()
                                user_metric_dict[metric][uid] = ndcg
                        else:
                            raise ValueError(f"Unsupported metric: {metric}")
        eval_results = {metric: np.mean(values) for metric, values in eval_results.items()}
        eval_msg = " - ".join([f"{metric}: {value:.4f}" for metric, value in eval_results.items()])
        logger.info(f"{behavior} test results - {eval_msg}")
        if len(user_metric_dict[self.metric_list[0]]) > 0:
            # Save user-level metrics
            save_path = os.path.join(
                self.result_dir,
                f"result-{self.test_task}",
                f"user_level_metrics_behavior_{behavior}.json"
            )
            ensure_dir(os.path.dirname(save_path))
            user_metric_list: dict[str, list[float]] = {}
            sorted_uids = sorted(user_metric_dict[self.metric_list[0]].keys())
            for m in self.metric_list:
                user_metric_list[m] = [user_metric_dict[m][uid] for uid in sorted_uids]
                assert len(user_metric_list[m]) == len(data_loader.dataset), "User-level metric length should match dataset length."
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(user_metric_list, f, indent=4)
            logger.info(f"Saved user-level metrics to {save_path}.")
        return eval_results

    def test(self) -> list[dict[str, float]]:
        results = []
        merge_results = {m: 0.0 for m in self.metric_list}
        total = 0
        for i, behavior in enumerate(self.behaviors):
            if isinstance(self.model, MBHT) and behavior != self.target_behavior:
                continue
            result = self.test_single_behavior(self.loaders[i], behavior)
            result['eval_type'] = f"Behavior {behavior}"
            results.append(result)
            for m in self.metric_list:
                assert m in result, f"Metric {m} not found in results for behavior {behavior}."
                merge_results[m] += result[m] * len(self.loaders[i].dataset)
            total += len(self.loaders[i].dataset)
        for m in merge_results:
            merge_results[m] /= total
        merge_results['eval_type'] = "Merged Behavior"
        results.append(merge_results)
        return results

    @staticmethod
    def _load_json_if_exists(path: str) -> dict:
        if path == "":
            return {}
        if not os.path.exists(path):
            logger.warning(f"Mapping file not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _decode_model_item_id(model_item_id: int, dataset: SMBDisDataset) -> int:
        # model item id 0 is padding index; valid items start from 1.
        if model_item_id <= 0:
            return -1
        if dataset.diff:
            return (model_item_id - 1) % dataset.num
        return model_item_id - 1

    def _build_submission_view_for_behavior(self, behavior: str) -> SMBDisDataset:
        # Do not filter by true target behavior. We create one inference view per behavior condition.
        behavior_idx = self.behaviors.index(behavior)
        view_dataset = copy.copy(self.base_test_data)
        view_dataset.inter_data = []
        for sample in self.base_test_data.inter_data:
            new_sample = sample.copy()
            new_sample["inter_behaviors"] = sample["inter_behaviors"].copy()
            if len(new_sample["inter_behaviors"]) > 0:
                new_sample["inter_behaviors"][-1] = behavior_idx
            new_sample["behavior"] = behavior_idx
            view_dataset.inter_data.append(new_sample)
        view_dataset.target_behavior = behavior
        return view_dataset

    def _collect_topk_candidates(
        self,
        loader: DataLoader,
        behavior: str,
        topk: int,
    ) -> dict[int, dict[str, str | list[int] | list[float]]]:
        results: dict[int, dict[str, str | list[int] | list[float]]] = {}
        with torch.no_grad():
            for batch, _ in get_tqdm(loader, desc=f"Submission collect - {behavior}"):
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                if "uid" not in batch:
                    raise ValueError("Submission export requires --add_uid.")

                uids = batch["uid"].detach().cpu().numpy().tolist()
                scores: np.ndarray = self.model.full_sort_predict(batch).detach().cpu().numpy()

                for i, uid in enumerate(uids):
                    row_scores = scores[i].copy()
                    if row_scores.shape[0] <= 1:
                        continue
                    row_scores[0] = -np.inf  # padding index

                    candidate_k = min(max(topk * 3, topk), row_scores.shape[0] - 1)
                    if candidate_k <= 0:
                        continue

                    candidate_idx = np.argpartition(-row_scores, candidate_k - 1)[:candidate_k]
                    candidate_idx = candidate_idx[np.argsort(-row_scores[candidate_idx])]

                    dedup_item_ids = []
                    dedup_scores = []
                    seen_items = set()
                    for model_item_id in candidate_idx.tolist():
                        item_id = self._decode_model_item_id(model_item_id, loader.dataset)
                        if item_id < 0 or item_id in seen_items:
                            continue
                        seen_items.add(item_id)
                        dedup_item_ids.append(item_id)
                        dedup_scores.append(float(row_scores[model_item_id]))
                        if len(dedup_item_ids) >= topk:
                            break

                    if len(dedup_item_ids) > 0:
                        results[int(uid)] = {
                            "behavior": behavior,
                            "items": dedup_item_ids,
                            "scores": dedup_scores,
                        }

        return results

    def _export_submission_file(self):
        if not self.add_uid:
            raise ValueError("Please enable --add_uid with --export_submission.")
        if self.submission_topk <= 0:
            raise ValueError("submission_topk must be positive.")

        uid_map_file = self.uid_map_file
        item_map_file = self.item_map_file
        if uid_map_file == "":
            uid_map_file = os.path.join(
                self.data_path,
                self.dataset_name,
                f"{self.dataset_name}.uid_to_original_session.json",
            )
        if item_map_file == "":
            item_map_file = os.path.join(
                self.data_path,
                self.dataset_name,
                f"{self.dataset_name}.item_to_raw_job.json",
            )

        uid_map = self._load_json_if_exists(uid_map_file)
        item_map = self._load_json_if_exists(item_map_file)

        by_uid: dict[int, list[dict[str, str | list[int] | list[float]]]] = {}
        for behavior in self.behaviors:
            behavior_dataset = self._build_submission_view_for_behavior(behavior)
            behavior_loader = DataLoader(
                behavior_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=TraditionalTestCollator(),
                drop_last=False,
                num_workers=0,
            )
            behavior_predictions = self._collect_topk_candidates(
                behavior_loader,
                behavior,
                self.submission_topk,
            )
            for uid, prediction in behavior_predictions.items():
                by_uid.setdefault(uid, []).append(prediction)

        submission_rows = []
        for uid in sorted(by_uid.keys()):
            candidates = by_uid[uid]
            if len(candidates) == 0:
                continue

            best_prediction = max(
                candidates,
                key=lambda x: x["scores"][0] if len(x["scores"]) > 0 else float("-inf"),
            )
            item_ids = best_prediction["items"][:self.submission_topk]
            raw_job_ids = [item_map.get(str(item_id), item_id) for item_id in item_ids] if len(item_map) > 0 else item_ids

            # uid in dataset starts from 1 when add_uid=True.
            uid_zero_based = str(uid - 1)
            session_id = uid_map.get(uid_zero_based, uid_map.get(str(uid), uid_zero_based))
            applies_for = str(best_prediction["behavior"]).lower() == self.apply_behavior_label.lower()

            row = {
                "session_id": session_id,
                "job_ids": json.dumps(raw_job_ids, ensure_ascii=False),
                "applies_for": bool(applies_for),
            }
            for rank in range(self.submission_topk):
                row[f"job_id_{rank + 1}"] = raw_job_ids[rank] if rank < len(raw_job_ids) else ""
            submission_rows.append(row)

        output_file = self.submission_file
        if output_file == "":
            output_file = os.path.join(self.result_dir, f"submission-{self.test_task}.csv")

        output_parent = os.path.dirname(output_file)
        if output_parent != "":
            ensure_dir(output_parent)

        fieldnames = [
            "session_id",
            "job_ids",
            "applies_for",
        ] + [f"job_id_{i + 1}" for i in range(self.submission_topk)]

        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(submission_rows)

        logger.success(f"Submission saved to {output_file} with {len(submission_rows)} rows.")

    def invoke(
        self,
        # global arguments
        seed: int,
        backbone: str,
        base_model: str,
        output_dir: str,
        result_dir: str,
        # dataset arguments
        data_path: str,
        tasks: str,
        test_task: str,
        dataset: str,
        index_file: str,
        max_his_len: int,
        add_uid: bool,
        # training arguments
        optim: str,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        logging_step: int,
        weight_decay: float,
        patience: int,
        metrics: str,
        wandb_run_name: str,
        only_test: bool,
        export_submission: bool,
        submission_file: str,
        submission_topk: int,
        uid_map_file: str,
        item_map_file: str,
        apply_behavior_label: str,
        *args,
        **kwargs,
    ):
        """
        Train the SMB decoder using the provided arguments.
        """
        # Implementation of the training logic goes here.
        set_seed(seed)
        if not only_test:
            import wandb
            wandb.init(
                project=self.parser_name(),
                config=self.param_dict,
                name=(
                    wandb_run_name
                    if wandb_run_name != "default"
                    else output_dir.split("checkpoint/SMB-recommender/")[-1]
                ),
                dir=f"runs/{self.parser_name()}",
                job_type="train",
                reinit="return_previous",
                notes=f"Training SMB recommender on {data_path} with base model {base_model}",
            )
        ensure_dir(output_dir)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning("Unused parameters:", args, kwargs)
        # backbone used for SMB recommendation model name
        config_cls: type[Config] = eval(f"{backbone}Config")
        config = config_cls.from_pretrained(base_model)

        train_data, valid_data = load_SMBDis_datasets(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            tasks=tasks,
            add_uid=add_uid,
        )
        self.target_behavior = valid_data.target_behavior
        valid_data = valid_data.filter_by_behavior(self.target_behavior)
        first_dataset: SMBDisDataset = train_data.datasets[0]
        num_items = first_dataset.num_items
        num_users = first_dataset.num_users
        self.behaviors = first_dataset.behaviors
        if backbone == 'MBHT':
            train_data = ConcatDataset([d.filter_by_behavior(self.target_behavior) for d in train_data.datasets])
        logger.info(f"Number of items: {num_items}")
        logger.info(f"Training data size: {len(train_data)}")

        if isinstance(first_dataset, SMBDisUserLevelDataset):
            logger.info("Using user-level collator for training.")
            train_collator = TraditionalUserLevelCollator()
        else:
            train_collator = TraditionalCollator()
        eval_collator = TraditionalTestCollator()
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_collator,
            num_workers=4,
        )
        eval_loader = DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=eval_collator,
            drop_last=False,
            num_workers=4,
        )

        model_cls: type[SeqModel] = eval(backbone)
        self.model = model_cls(config, n_items=num_items, n_users=num_users, max_his_len=max_his_len, target_behavior_id=first_dataset.target_behavior_index + 1, n_behaviors=len(self.behaviors))
        logger.info(self.model)

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        if not only_test:
            from SeqRec.trainers.SMBRec import Trainer
            trainer = Trainer(
                model=self.model,
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
                optim=optim,
                lr=learning_rate,
                weight_decay=weight_decay,
                epochs=epochs,
                logging_step=logging_step,
                output_dir=output_dir,
                patience=patience,
                metrics=metrics.split(","),
            )

            trainer.train()
            logger.info("Training completed successfully.")
            wandb.finish()
        else:
            logger.info("Skipping training as only_test is set to True.")
            self.model.to(self.device)

        self.metric_list = metrics.split(",")
        test_data = load_SMBDis_test_dataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            test_task=test_task,
            add_uid=add_uid,
        )
        self.base_test_data = test_data
        self.datasets: list[SMBDisDataset] = []
        for behavior in self.behaviors:
            self.datasets.append(test_data.filter_by_behavior(behavior))
            self.info(f"Loaded dataset for behavior {behavior} with {len(self.datasets[-1])} samples.")
        self.loaders = [
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=eval_collator,
                drop_last=False,
                num_workers=0
            ) for dataset in self.datasets
        ]
        state_dict = torch.load(output_dir + '/best_model.pth', map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.result_dir = result_dir
        self.test_task = test_task
        self.export_submission = export_submission
        self.submission_file = submission_file
        self.submission_topk = submission_topk
        self.uid_map_file = uid_map_file
        self.item_map_file = item_map_file
        self.apply_behavior_label = apply_behavior_label
        self.batch_size = batch_size
        self.add_uid = add_uid
        self.data_path = data_path
        self.dataset_name = dataset

        results = self.test()
        logger.success("======================================================")
        logger.success("Results:")
        for res in results:
            logger.success("======================================================")
            logger.success(f"{res['eval_type']} results:")
            for m in res:
                if isinstance(res[m], float):
                    logger.success(f"\t{m} = {res[m]:.4f}")
        logger.success("======================================================")
        ensure_dir(self.result_dir)
        result_file = os.path.join(self.result_dir, f"result-{self.test_task}.json")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.success(f"Results saved to {result_file}.")
        if self.export_submission:
            self._export_submission_file()
