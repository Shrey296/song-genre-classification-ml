In this project, I worked with a Spotify playlist dataset that was extracted directly from Spotify using a tool called Exportify. The dataset was used to build a machine learning model that predicts the broad genre of a song. The workflow followed the standard stages of data preprocessing, feature engineering, model training, and evaluation.

I first removed unnecessary or missing values, particularly in the “Genres” column, since music contains a wide variety of genre labels. Because the original genre names were highly specific and sometimes inconsistent, I grouped them into broader and more meaningful categories such as Rap/HipHop, Rock, Pop, Indie, Anime/JPop, R&B/Soul, and Other. This feature engineering step helped simplify the classification problem and allowed the model to better learn patterns across similar genres.

Next, I selected twelve numerical audio features provided by Spotify, including Danceability, Energy, Key, Mode, Loudness, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, and Time Signature. These features were standardized using StandardScaler to ensure uniform scaling and prevent any single feature from dominating due to differences in measurement units.

I then trained a Logistic Regression model, which is a foundational machine learning algorithm widely used for classification tasks. The model performance was then evaluated using accuracy, a classification report and a confusion matrix. A feature importance chart was also plotted to identify which audio characteristics had the greatest influence on predictions. The model achieved an accuracy of approximately 69.3%. Based on the evaluation results, the model demonstrated strong predictive capability, indicating that the selected features and preprocessing steps were effective for genre classification.

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/cb9e5103-a349-4e22-994a-457f5121a2f8" />

<img width="700" height="700" alt="image" src="https://github.com/user-attachments/assets/5e72855b-8f1d-45b5-8f3f-c4875940f34f" />
