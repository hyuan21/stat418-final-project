# Final Presentation — Speaker Script

**Total length: ~10 minutes.**
**Style: short sentences, simple words, native speaker friendly.**

---

## Slide 1 — Title  (45 seconds)

Hi everyone. My name is Hanzhang Yuan.

Today I will talk about my final project.

The project is called **"Predicting NBA Playoff Underperformance."**

Here is the idea in one sentence:

> Before a playoff game starts, can a model predict if an NBA player will play worse than his regular-season level?

I built two things you can try right now:
- a **Web App** on Google Cloud Run
- a **REST API**, also on Cloud Run

Both links are on the slide. They are live.

Let me show you how I built it.

---

## Slide 2 — Overview & Motivation  (1 minute)

Playoff basketball is different from the regular season.

Defenses get tighter. Rotations get shorter. Every star gets scouted.

So **some stars fade in the playoffs. Others rise.**

I wanted to know: can a model tell the difference **before** the game?

This is my research question, top-left in blue.

A few key choices, on the right:

- **Binary** — yes or no, did the player underperform?
- **Player-game** — one row is one player in one game. Not one player in a whole season.
- **22 seasons** of data — from 2003-04 to 2024-25.

And four objectives:
1. Scrape 22 seasons.
2. Build features that don't leak future data.
3. Compare models and explain them.
4. Deploy a real App and API.

I did all four. Let me walk through them.

---

## Slide 3 — Data Collection & Preprocessing  (1.5 minutes)

I pulled data from the **NBA Stats API**, using the `nba_api` Python library.

This gave me every player's box score from every game, over 22 seasons.

**The numbers, on the right:**
- About **25,000** API requests
- **575,801** player-game rows in total
- After filtering to playoffs only, **about 30,000** rows for modeling

**The pipeline, on the left, has four steps:**

1. **Scrape.** Slow and careful. I added a 1.2-second delay between requests, retries on failure, and a disk cache. So if the network broke, I could resume without losing data.

2. **Parse.** I turned the raw JSON into a clean Parquet table. I computed a basic basketball metric called Hollinger Game Score.

3. **Aggregate.** I built per-season averages for each player. I also built per-season defensive stats for each team.

4. **Engineer.** This is the hardest step. I built **45 features**, including:
   - z-scores normalized **per season**, so a 25-point average in 2005 means the same thing as in 2024
   - expanding-window playoff history, so the features for a Game 5 only use games 1 to 4

**Challenges, at the bottom in dark blue:**
- The league plays differently in 2024 than in 2005. I fixed it with per-season z-scores.
- The NBA API kept dropping connections. I patched the headers and added retries.
- Game Scores are noisier for stars than for bench players. I used a relative 20% threshold, not a fixed number.

---

## Slide 4 — Model Development & Evaluation  (1.5 minutes)

I trained three models on the same features. Here they are.

**Logistic Regression**, baseline. **Random Forest**, middle. **XGBoost**, the one I deployed.

Look at the test AUCs. All three are about **0.65**. Almost the same.

So why did I deploy XGBoost? Two reasons:
- It works better with SHAP for explanations.
- It is more robust when a player has missing features.

The F1 scores are also similar — around 0.63.

**On the bottom-left, the top SHAP features.**

The most important one is **last game's Game Score**. That makes sense — momentum matters.

Then minutes per game, home or away, career playoff history.

Nothing surprising. The model picks up basketball common sense.

**On the bottom-right, the six feature families.**

Player baseline, playoff history, game context, opponent strength, matchup history, in-series momentum.

**The blue bar at the bottom is the most important sentence on this slide:**

> Target definition mattered more than model choice. Cleaning the target gave me 3.8 percentage points. Switching models gave me 0.1.

This was the most surprising lesson of the whole project. **The problem setup, not the algorithm, was the bottleneck.**

---

## Slide 5 — Solution Architecture  (1.5 minutes)

The system has two flows.

**Offline training, on the top.** This runs once.

NBA Stats API → cached raw data → feature table → model training → model artifacts.

I built each step as its own Python module. So I can rerun any step alone.

**Online inference, on the bottom.** This runs every time someone visits the App.

User → Streamlit App → REST call → Flask API on Cloud Run → output with risk score, predicted Game Score, and SHAP explanation.

The two flows are separate. The training data never touches the live API. The live API never modifies the training data. That is a clean architecture pattern.

**On the right, the tech stack.**

- Data: `nba_api`, pandas, pyarrow
- Model: scikit-learn, XGBoost, Optuna, SHAP
- API: Flask plus flask-restx, with Swagger docs
- App: Streamlit
- Deploy: Docker on Google Cloud Run
- Quality: pytest and GitHub Actions

Everything is containerized. Anyone can clone the repo, run `docker compose up`, and have the full system running locally in about three minutes.

---

## Slide 6 — Application Features Walkthrough  (1.5 minutes)

The App has **two modes**.

**Replay mode, on the left.** You pick a real playoff game from 2023 or 2024. The model gives its pre-game prediction. The App then shows what actually happened.

**What-if mode, on the left, bottom.** You pick any player, any opponent, any game number. Even fake matchups. Use this to predict 2025–26 playoff games that haven't happened yet.

**The screenshot in the middle is from Replay mode.**

I picked **Jokić versus the LA Clippers, Game 1 of the 2024-25 playoffs.**

- The model predicted **37%** chance of underperform — mixed signal.
- The model predicted a Game Score of **27.9**.
- Jokić's regular-season average was **30.1**.

Then look at what actually happened:
- He **did not** underperform.
- His actual Game Score was **27.4**.
- So the model **called it correctly.**

And on the bottom, the SHAP explanation:
- Weak previous game pushed risk **up**.
- Lower minutes pushed risk **down**.
- Strong home-playoff history pushed risk **down**.

The user sees the **why**, not just a number.

---

## Slide 7 — Challenges, AI, Lessons, Future  (1.5 minutes)

**Top four challenges, on the left:**

1. **Data leakage.** Easy mistake to make in time-series. I wrote seven pytest assertions before writing the pipeline. They caught two real bugs.

2. **API connection resets.** The NBA Stats API kept dropping me. I added patched headers, retries, and disk caching. After that, I could resume scraping at any time.

3. **Heteroskedastic baselines.** A 1-point drop for Jokić is noise. For a bench player it is a 25% slump. I switched to a relative 20% threshold. This gave me almost 4 percentage points of AUC.

4. **Identical predictions bug.** At one point the API returned the same answer for every input. I tracked it down to the feature lookup module. I rewrote it to use the real training data.

**On the right, my AI usage.**

I used an AI coding assistant during this project. I treated it like a smarter Stack Overflow.

**All the design decisions — features, target, evaluation, architecture — were mine.** I would draft the design, then ask the AI for an implementation, then review the code line by line and modify it.

The AI was most useful for **boilerplate and test scaffolding.** It was least useful when I gave it open-ended questions without my own context first.

**At the bottom:**

- **Key lesson:** A clean target definition and good leakage control mattered more than choosing a complex model.
- **Future work:** scheduled weekly scrapes, injury features, series-level predictions, better calibration, live 2025–26 predictions.

---

## Closing  (15 seconds)

That's the project.

The links to the live App and API are on Slide 1.

The full code, README, and the writeup are on my GitHub.

Thank you. I'm happy to take any questions.

---

# Speaker Notes

## Timing Cheat Sheet

| Slide | Topic | Target Time |
|---|---|---|
| 1 | Title | 0:45 |
| 2 | Overview | 1:00 |
| 3 | Data | 1:30 |
| 4 | Model | 1:30 |
| 5 | Architecture | 1:30 |
| 6 | App walkthrough | 1:30 |
| 7 | Reflection | 1:30 |
| — | Closing + buffer | 0:45 |
| **Total** | | **10:00** |

## Delivery Tips

- **Pause** after each slide title. Let the audience focus on the slide before you start talking.
- **Read the big numbers slowly.** 575,801 is "five hundred seventy-five thousand, eight hundred one." 0.65 is "zero point six five" or just "point six five."
- **Don't memorize.** Use this as a guide. The slides do the heavy lifting.
- **Point at the screen** when you mention a region of the slide ("here on the left", "the blue bar at the bottom").
- If you run short on time, **cut slide 5 details first** (architecture is mostly visual).
- If you have extra time, **expand slide 4** (the +3.8 pp lesson is the most interesting story).

## Hard Words — Practice Pronunciation

- **Hollinger** = HALL-in-jer
- **heteroskedastic** = HET-er-oh-skuh-DAS-tik (skip this word in speech; say "noisy baselines" instead)
- **Optuna** = op-TOO-nuh
- **expanding-window** = pause between each word
- **Jokić** = YOH-kitch
- **Streamlit** = STREAM-lit
- **gunicorn** = GOO-ni-corn (don't say this on slides — too technical)

## Q&A Preparation

Likely questions + short answers:

**Q: Why XGBoost over Logistic Regression if AUC is the same?**
A: XGBoost handles missing features better and works better with SHAP. The AUC tie tells us the data is the bottleneck, not the model.

**Q: Is 0.65 AUC any good?**
A: For single-game NBA player performance, yes. Research-grade AUC in this area is 0.58 to 0.65. Single games have a lot of randomness — rim luck, foul trouble, in-game adjustments.

**Q: How do you prevent data leakage?**
A: Three things. (1) Time-ordered train/val/test split. (2) Expanding-window features that only use past games. (3) Seven pytest assertions that verify no feature uses future data.

**Q: How much was AI?**
A: AI helped with boilerplate code and test scaffolding. All design choices were mine. I reviewed every generated line before committing.

**Q: Can it predict ongoing 2025-26 playoffs?**
A: Yes — that's the What-if mode. The model uses 2025-26 regular-season stats as the player baseline.

**Q: What would you do differently?**
A: Add injury status as a feature. That's the biggest missing signal — players who play through injuries will underperform mechanically.

**Q: Why didn't a bigger model help?**
A: Probably because single-game outcomes have a high noise floor. To get past 0.65 AUC, I would need new information — injuries, lineup combinations, defensive matchups — not a fancier algorithm.
