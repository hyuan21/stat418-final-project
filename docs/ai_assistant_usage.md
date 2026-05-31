# AI Assistant Usage — STAT 418 Final Project

Per the project rubric, this document records how AI coding assistants were used throughout development of the NBA Playoff Underperformance Predictor.

## Assistants Used

- **Claude (Anthropic)** — primary collaborator, used through Cowork (the Claude desktop app). Used for architecture brainstorming, code generation, debugging, documentation, and slide preparation.

## Tasks Where AI Was Used

### 1. Project Scoping & Architecture
The initial proposal framed the problem at player-season granularity, which would have produced only ~2,400 training samples. Claude was used to brainstorm alternatives and ultimately settled on **player-game granularity with z-score season normalization**, which produced ~45,000 samples and naturally handled NBA era differences.

**Particularly helpful prompt pattern:** "Here is my proposal. Brainstorm 4 ways to expand the sample size, with concrete tradeoffs for each. Then recommend one and explain why." This kind of decision-framing prompt produced far better outputs than asking "make my project bigger".

### 2. Feature Engineering Design
Claude helped enumerate the 6 families of features (player baseline, player history, game context, opponent, player-opponent matchup, series momentum) and design the time-window-locked computation that prevents data leakage. This is the most subtle part of the project and the one most likely to be silently wrong without AI review.

### 3. Code Generation
- nba_api wrapper with rate limiting + caching
- Feature-engineering pipeline with strict expanding-window computation
- Flask API scaffolding with OpenAPI auto-docs
- Streamlit App layout
- pytest test cases for the API and game-score formula
- GitHub Actions YAML for CI

### 4. Debugging
*(To be filled in as development proceeds.)*

### 5. Documentation
This README, the writeup, and the README files inside each subdirectory were drafted with Claude and edited by the author.

## Where AI Output Required Significant Modification

*(To be filled in as development proceeds. Expected areas: rate-limit handling for nba_api in practice, SHAP plot rendering inside Streamlit, Cloud Run cold-start tuning.)*

## Lessons Learned

1. **AI is excellent at the "shape" of a problem but you have to define it.** Asking Claude to "build me a machine learning project" produces generic mush. Asking it to "expand player-season prediction to player-game prediction while preventing leakage" produces a sharp, useful answer. The author's job is to frame the question; AI's job is to fill in the technical detail.

2. **AI catches design mistakes earlier than humans do.** Originally the input feature `opponent_team_id` was specified as a categorical variable. Claude pointed out that a team's defensive identity changes from season to season, so this would overfit to team brands; it recommended using continuous defensive stats (def_rating, pace, opponent eFG%) instead. This is the kind of correction that often only surfaces in code review, far later in the process.

3. **Always ask AI to explain *why*, not just *what*.** When Claude proposes a design, asking "why this rather than X?" forces it to articulate the tradeoffs explicitly — and sometimes reveals that the alternative is actually better for your specific constraints.

4. **AI is great at consistency.** Once a project structure was decided, Claude produced README files in matching format for every subdirectory in seconds, with consistent terminology and cross-references — something tedious and error-prone to do by hand.

5. **Verify everything that touches the data.** The leakage-prevention code in particular was reviewed line by line — AI is helpful but can make subtle off-by-one errors in time-window logic that ruin the project's validity.
