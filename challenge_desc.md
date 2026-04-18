Challenge context
When candidates search for their ideal job, their journey through a job board often follows a sequential path, moving from one job listing to the next based on their interests, preferences, and skills. This exploration is similar to a traveler navigating through a series of destinations, where each opened job represents a new point of interest.

The goal is to predict which job a candidate is most likely to open next by using historical data from their previous interactions with job postings.

Typically, when a candidate first visits a job board, no information is available about their background to make personalized recommendations. To address this, many websites ask candidates to upload their resume or fill in forms with personal details. However, this approach often leads to loss of attention or interest, as candidates may feel overwhelmed and simply leave the website before exploring job opportunities.

To overcome this issue, we propose a solution similar to platforms like YouTube or Amazon, where we recommend the best jobs based purely on navigation behavior without requiring the candidate to sign in or provide any upfront information. By analyzing the sequence of job listings the candidate views and applies to, we can intelligently suggest relevant opportunities, creating a more seamless and engaging job search experience.

The hidden challenge in this problem lies in the shifting interests of candidates. A candidate may initially be interested in a certain set of job opportunities, but after viewing them, his preferences might shift to a different set of jobs.

Challenge goals
Challenge Objective
Objective: Use Online Machine Learning and Collaborative Filtering methods (or any other method) to predict, based on the browsing history:
The top 10 jobs a candidate is most likely to explore next.
The candidate’s next action. It can be view or apply.
Evaluation Metrics
The metrics used in this challenge to rank participants are:

Mean Reciprocal Rank (MRR): Measures the average of the reciprocal ranks of the correct job within the top 10 predicted jobs. Higher MRR indicates the correct job is ranked higher in the recommendations.
Accuracy: Evaluates how well the model predicts whether the candidate’s next action will be an apply or view, as indicated by the “action” variable.
The final score is a weighted combination of these two metrics:

70% MRR
30% Accuracy on the “action” variable
MRR Examples
If the correct job is ranked 1st, the reciprocal rank is 1.0.
If the correct job is ranked 2nd, the reciprocal rank is 0.5.
If the correct job is ranked 3rd, the reciprocal rank is 0.333.
If the correct job is not in the top 10, the reciprocal rank is 0.
Notes
Predictions rely on the sequence of jobs viewed and applied to, without using profile-specific data such as education or skills.
Highlights that candidates’ interests may evolve during their search, necessitating approaches like Online Machine Learning and Non-Stationary Multi-Armed Bandits to adapt dynamically.
Data description
The dataset provided for this challenge consists of two main components: job listings and the candidate’s job exploration history. The dataset is stored in CSV format and is structured as follows:

Job Listings:
Job ID: A unique identification number for each job listing.
Job Text: A text that contains the description of the job offer. This text is not perfect and contains some realistic data imperfections.
Candidate Interaction History (X train/test):
Session ID: A unique identification number for each job search session.
Job IDs: A sequence of job IDs that the candidate has opened during the session, representing their browsing history on the job board.
Actions: The action performed by the candidate to the corresponding job. Could be “view” or “apply”.
Target Job (y train/test):
Session ID: A unique identification number corresponding to each session.
Target Job ID: The ID of the next job the candidate will view after interacting with previous jobs in the sequence.
Action: The action performed by the candidate to the corresponding job. Could be view or apply.
The output format for predictions should include:

A list of 10 possible jobs for each session, ranked by the likelihood that the candidate will open them next.
An action indicating whether the candidate’s next action will be an “apply” or a “view”.
Each session ID should be associated with the list of 10 predicted job IDs in descending order of probability, along with the single boolean corresponding to applies_for.

Dataset Statistics:
The dataset contains approximately 18K candidate sessions for train and test splits.
These sessions involve approximately 22K unique job listings and 10K unique profile sessions.
Within the training and testing sets:

Some long profile sessions have been converted into multiple smaller sessions without overlaps.
There will be no overlap between the train and test sets (profile sessions in the training set will never be related to the testing ones).
Notes:
It is important to account for potential imbalances in the dataset, where certain types of jobs may be opened more frequently than others, which could impact model performance.