### *🏀 Scouting System for NBA Players*

🚀 Live Demo: Check out the interactive app here --> https://scouting-system-for-nba-players-9gcv5t5mgrmv6nbfjaxjr6.streamlit.app

### *Leveraging Machine Learning to Redefine Player Performance Evaluation*

In the modern era of sports, where the margin for error has narrowed and the quest for individual excellence has intensified, traditional box scores are no longer sufficient. This project introduces a data-driven approach to player scouting, moving beyond simple box-score analysis by implementing a **Customized Efficiency Rating (CER)** and advanced Machine Learning algorithms.

The primary objective is to classify NBA All-Star players (from the 2019 to 2024 seasons) into three distinct performance tiers: **MVP**, **All-NBA**, and **All-Star** calibers.

## 🚀 Key Features
* **Customized Efficiency Rating (CER):** A proprietary metric developed to reflect on-court impact more accurately than traditional statistics.
* **Logarithmic Games-Played Adjustment:** CEM scores are refined using `np.log(games_played)` to balance peak performance with season-long durability.
* **Hybrid ML Pipeline:** Integration of Unsupervised Learning (**K-Means**) for cluster discovery and Supervised Learning (**Random Forest**) for precise classification.
* **Robust Validation:** Deployment of K-Fold and Leave-One-Out (LOO) cross-validation techniques to ensure model stability and generalizability.

## 🛠 Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Algorithms:** Random Forest Classifier, K-Means Clustering, Logistic Regression (for baseline comparison)
* **Optimization:** Hyperparameter Tuning via Grid Search (over 600 fits)

## 📊 The CER Metric & Methodology

The formula is designed to balance offensive production, efficiency, and defensive impact:

Primary Contributors (75%): Points (35%), Rebounds (20%), and Assists (20%) form the core of the rating.

Efficiency & Defense (25%): Field Goal % (10%), Free Throw % (5%), Steals (5%), and Blocks (5%) are integrated to reward two-way performance and efficient scoring.

Turnover Penalty (-10%): A dedicated penalty for Turnovers (TO) to account for ball security and offensive reliability.

Sustainability Factor: By multiplying the raw score with the natural logarithm of Games Played, the metric rewards players who maintain high performance levels throughout the entire season, acknowledging that durability is a key component of an MVP-caliber season.

To bridge the gap in traditional analytics, this project utilizes a weighted formula that emphasizes efficiency and versatile contributions:

$$CEM_{raw} = (0.35 \cdot PPG_{n}) + (0.20 \cdot RPG_{n}) + (0.20 \cdot APG_{n}) + (0.10 \cdot FG\%_{n}) + (0.05 \cdot (SPG_{n} + BPG_{n} + FT\%_{n})) - (0.10 \cdot TO_{n})$$

$$Final\ CEM = CEM_{raw} \times \ln(Games\ Played)$$


*The raw score is further adjusted by the natural logarithm of games played to account for the "diminishing returns" of volume over duration.*

## 🧠 Experimental Results & Validation
The model demonstrates exceptional robustness and predictive power:

* **Model Accuracy:** Achieved a **93.3% Mean K-Fold Cross-Validation** score and a **96.6% Leave-One-Out (LOO)** accuracy.
* **Precision Performance:** Attained a **1.00 Precision for MVP Caliber** classification, effectively identifying top-tier candidates without false positives.
* **Optimal Configuration:** Through Hyperparameter Tuning, the Random Forest model was optimized at `n_estimators=50` and `max_features='sqrt'`.

## 📈 Conclusion
This project proves that Machine Learning can significantly enhance sports analytics by identifying performance patterns that are often missed by the naked eye. The success of the CER metric in differentiating player tiers demonstrates the potential for these algorithms to positively influence scouting and team management strategies in professional basketball.
