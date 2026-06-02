# AI Assistant Usage — STAT 418 Final Project

Per the project rubric, this document records how AI coding assistants were used throughout development of the NBA Playoff Underperformance Predictor.

## Tools

I used an AI coding assistant (a large-language-model based chat tool) as a coding helper — similar in role to Stack Overflow or a senior engineer to bounce ideas off. **All architectural decisions, modeling choices, target definition, and evaluation methodology in this project are my own.** AI was consulted only after I had drafted the design for each component.

## How I used it, by phase

### 1. Project Scoping & Problem Framing
My initial proposal framed the problem at player-season granularity, which would have produced only ~2,400 training samples. I used the AI to brainstorm alternatives and pressure-test my reasoning. After comparing options I chose **player-game granularity with z-score season normalization**, which produced ~45,000 samples and naturally handled NBA era differences. The AI's role here was to help me see the tradeoffs more clearly — the decision was mine.

**Particularly helpful prompt pattern:** I would describe a design I was considering and ask "what's the strongest argument against this?" That kind of adversarial prompt was more useful than open-ended "build me a machine learning project" prompts, which produced generic output.

### 2. Feature Engineering Review
I designed the six feature families (player baseline, player history, game context, opponent, player-opponent matchup, series momentum) and the time-window-locked computation that prevents data leakage. I then asked the AI to review my design and draft pytest assertions for it before I wrote the implementation. The resulting seven assertions caught two real bugs in my code — both off-by-one errors in expanding-window computation that I would have missed without explicit tests.

### 3. Code Generation
I used the AI to generate boilerplate code for components I had already designed:
- nba_api wrapper with rate limiting + caching (I specified retry strategy + cache layout)
- Flask API scaffolding with OpenAPI auto-docs (I specified endpoints + request/response schemas)
- Streamlit App layout (I specified the two-mode design)
- GitHub Actions YAML for CI (I specified what to run)

In every case I reviewed the output line by line and modified or rewrote portions before committing.

### 4. Debugging
The most useful debugging moment was diagnosing a bug where my API returned identical predictions for every input. I described the symptom to the AI, which pointed me to inspect the feature lookup module. I confirmed the issue (it was returning NaN for nearly every column), then rewrote the module from scratch to query the real training table.

### 5. Documentation
This README, the writeup, and the per-directory README files were drafted with AI assistance and then edited by me for accuracy and tone.

## Where AI output required significant modification

1. **Feature lookup module.** The first generated version returned NaN columns, causing every API prediction to come back the same. I had to fully rewrite it to query `playoff_features.parquet` directly and recompute series-momentum from real games 1..N−1 at lookup time.

2. **Streamlit App data flow.** The first version of the App's Replay mode used a flat dropdown of every NBA player; users couldn't tell who actually played in a given season's playoffs. I added cascading dropdowns and an API endpoint that returns only real matchups for the selected (player, season).

3. **Underperform target definition.** The first AI-suggested target used a fixed 1.0 GS absolute threshold. I noticed this was statistically wrong (a 1.0 drop for Jokić is noise; for a bench player it is a 25% slump) and switched to a relative 20% threshold with a 1.0 GS floor. This single change added +3.8 percentage points to test AUC.

## Lessons learned

1. **AI accelerates the obvious parts of engineering and leaves the judgment work entirely to you.** AI was useful when I had already framed the problem clearly. It was actively unhelpful when I gave it open-ended questions without my own context.

2. **Adversarial prompts beat open-ended prompts.** Asking "what's wrong with this design?" surfaces more value than asking "design this for me."

3. **Tests before code is even more important with AI.** Because AI-generated code looks confident even when it's wrong, having pytest assertions ready before generation catches mistakes I might otherwise trust into production.

4. **Treat AI like a smart but overconfident intern.** Trust the direction, verify every output. The leakage assertions and the no-NaN feature-lookup tests in this repo exist precisely because I do not assume AI output is correct.

5. **The judgment work — what to predict, how to define the target, how to split the data, what counts as a useful feature — is the part that determines whether the project is good.** AI cannot do that part for me, and I learned to stop trying to make it.
