Abhinav
abhinavexist
•
it was all a dream.

RS2208 — Yesterday at 23:24
Lets give the best we can these 2 days
Plat Demon — Yesterday at 23:25
Sure!!
RS2208 — Yesterday at 23:25
I wanted to ask one thing, have anyone of you ever worked on something related to track 1?
Abhinav

 — Yesterday at 23:27
Not really , have a general idea of how things go but if you are asking that have implemented something concrete then no. Its our first time too.
RS2208 — Yesterday at 23:28
Yeah no worries. It’s my first time working something like that too. I guess we have to design our question model with SFT via synthetic data and use RL on the answer model
Abhinav

 — Yesterday at 23:29
yup Supervised fine tuning is the way to go.
RS2208 — Yesterday at 23:30
My only worry is that because it’s two models we have to train, we will have less time to experiment and train with
Track 2 is interesting, but involves more RL pipeline design
Abhinav

 — Yesterday at 23:34
yes, like each model on its own will take around 6-8 hrs to train
considering mid optimisations during training
RS2208 — Yesterday at 23:53
Yeah, SFT won’t take much time though. RL and data would
Anyways let’s discuss once before registration for the track and event
RS2208 — 08:56
Have any of you reached? 
Anime_Boy575 — 08:57
It will take us about 25 minutes
RS2208 — 09:10
Okay, i reached
I am inside the lecture hall b19
Anime_Boy575 — 09:16
Ok
We are at hauz khas
RS2208 — 09:31
You all reached?
Anime_Boy575 — 09:31
Yes we are exiting the station
We'll be there in a bit
RS2208 — 09:59
Your registrations are ongoing?
Abhinav

 — 10:00
we are in line for registration
Anime_Boy575 — 14:03
Image
Image
Image
Image
Can only use these models 
Image
Image
Image
Image
Anime_Boy575 — 14:12
Image
Anime_Boy575 — 14:27
https://dev.amd-ai-academy.com/
@Plat Demon
RS2208 — 14:29
t#6tK8SiDHDtLGRo
Plat Demon — 14:32
https://dev.amd-ai-academy.com/jupyter/jupyter-team-team5-260214085622-9245/lab/tree/AAIPL
Plat Demon — 14:55
# AMD AI Premier League (AAIPL) - Complete Guidelines
**IIT Delhi Competition Overview**

---

## Table of Contents

project_rules.md
16 KB
﻿
# AMD AI Premier League (AAIPL) - Complete Guidelines
**IIT Delhi Competition Overview**

---

## Table of Contents

1. [Competition Overview](#competition-overview)
2. [Match Structure](#match-structure)
3. [Scoring System](#scoring-system)
4. [Guidelines](#guidelines)
5. [Required Files & Examples](#required-files--examples)
6. [Topics](#topics)
7. [Possible Approaches & WhiteList Models](#possible-approaches--whitelist-models)
8. [Submission Requirements](#submission-requirements)
9. [Restrictions](#restrictions)
10. [Token & Time Limits](#token--time-limits)
11. [JSON Format Specifications](#json-format-specifications)
12. [Takeaways](#takeaways)

---

## 1. Competition Overview

### Visual Representation
🏆 **AMD AI Premier League (AAIPL)** 🏆

The competition features a match between two AI agent teams:

- **Team A** (Red Robot)
- **Team B** (Black Robot)

### Agent Roles
Each team consists of two agents:

1. **Q-Agent (Question Agent)** - Generates questions
2. **A-Agent (Answer Agent)** - Answers questions

---

## 2. Match Structure

### Game Format
The match is played in **2 Innings**:

#### **Inning 1:**

- **Team A:**
  - Q-Agent generates 20 Questions
  - A-Agent provides answers
  - Results: 15 answered correctly

- **Team B:**
  - Q-Agent generates questions
  - A-Agent provides answers

#### **Inning 2:**

- **Team A:**
  - Q-Agent generates 20 Questions
  - A-Agent provides answers
  - Results: 7 answered correctly

- **Team B:**
  - Q-Agent generates questions
  - A-Agent provides answers

### How It Works

- In each inning, teams alternate roles
- One team's Q-Agent generates questions for the opponent's A-Agent
- The A-Agent must attempt to answer all questions

---

## 3. Scoring System

### Scoring Logic

1. **Q-Agent Scoring:**
   - The Q-Agent scores points when the opponent's A-Agent **fails** to answer correctly
   - Points earned = Number of incorrect/unanswered questions

2. **A-Agent Scoring:**
   - The A-Agent scores points when it answers questions **correctly**
   - Points earned = Number of correct answers

### Example Scoring Breakdown

#### **Inning 1:**

- **Team A:**
  - Q-Agent: 5/20 (opponent answered 15, failed 5) → **5 points**
  - A-Agent: 15/20 (answered 15 correctly) → **15 points**

- **Team B:**
  - A-Agent: 5/20 (answered 5 correctly) → **5 points**
  - Q-Agent: 15/20 (opponent answered 5, failed 15) → **15 points**

#### **Inning 2:**

- **Team A:**
  - A-Agent: 7/20 (answered 7 correctly) → **7 points**
  - Q-Agent: 13/20 (opponent answered 7, failed 13) → **13 points**

- **Team B:**
  - Q-Agent: 7/20 (opponent answered 13, failed 7) → **7 points**
  - A-Agent: 13/20 (answered 13 correctly) → **13 points**

#### **Final Scores:**

- **Team A Total Score:** 5 + 15 + 7 + 13 = **40 points**
  - (Note: The slide shows 12, but calculation suggests different scoring method)
- **Team B Total Score:** 5 + 15 + 7 + 13 = **40 points**
  - (Note: The slide shows 28, but calculation suggests different scoring method)

**🥇 Team B Wins! 🥇**

*Note: The exact scoring calculation method should be verified with organizers as the slide shows Team A: 12, Team B: 28*

---

## 4. Guidelines

⚠️ **IMPORTANT:** This section contains visual guidelines displayed in the presentation.

**Key Points to Review:**

- Slide 4 contains important guideline images that need to be reviewed in the original presentation
- These guidelines provide critical information about competition rules and requirements

**Action Required:** Please refer to Slide 4 in the original PowerPoint to view the complete guidelines.

---

## 5. Required Files & Examples

### 📋 Necessary Files

You will need to work with the following files:

#### 1. **sample_answer.json**

- Details how `answers.json` should look like
- Provides the expected format for A-Agent responses

#### 2. **sample_questions.json**

- Details how `questions.json` should look like
- Provides the expected format for Q-Agent responses

#### 3. **topics.json**

- Contains the three main topics for the competition
- Defines the scope of questions that can be generated

#### 4. **topics_example.json**

- Consists of 1-3 examples from each topic
- Serves as reference for question and answer quality

---

## 6. Topics

### 📚 Competition Topics

The competition focuses on **FOUR** specific logical reasoning topics:

### Topic 1: **Syllogisms**

- Logical reasoning based on premises and conclusions
- Deductive reasoning patterns

### Topic 2: **Seating Arrangements**

- **Circular Seating Arrangements**
- **Linear Seating Arrangements**

**⚠️ IMPORTANT RESTRICTION:**

- **DO NOT include numeric style seating arrangement questions**
- Examples of prohibited questions:
  - "How many permutations of such arrangements are possible?"
  - Questions asking for counting/enumeration of arrangements
  - Any numeric calculation-based seating questions

### Topic 3: **Blood Relations and Family Tree**

- Relationship inference problems
- Family structure and connections
- Generational relationships

### Topic 4: **Mixed Series (Alphanumeric)**

- Pattern recognition in alphanumeric sequences
- Letter and number series
- Combined alphabetic and numeric patterns

---

## 7. Possible Approaches & WhiteList Models

### 🤖 WhiteList Models

You **MUST** use one of the following approved models for your final submission:

1. **Qwen/Qwen3-4B**
2. **Qwen/Qwen2.5-14B-Instruct**
3. **Unsloth/Llama-3.1-8B-Instruct**
4. **mistralai/Mistral-7B-Instruct-v0.3**
5. **microsoft/Phi-4-mini-instruct**
6. **google/gemma-3-12b-it**

### 🎯 Allowed Training Approaches

You may use the following techniques to improve your agents:

#### 1. **SFT (Supervised Fine-Tuning)**

- Train with your synthetic reasoning datasets
- Create custom training data for the four topics

#### 2. **Reinforcement Learning**

- Train agents through reward-based learning
- Optimize question difficulty and answer accuracy

#### 3. **Prompt Tuning**

- Optimize prompts for better performance
- Engineer effective instruction templates

#### 4. **Distillation**

- **You can use ANY teacher model** to guide the distillation process
- Create training data from larger/more capable models
- **⚠️ CRITICAL:** The **final version** of both Q-Agent and A-Agent **MUST be one of the WhiteList models**
- You cannot submit a non-whitelisted model, even if it was trained via distillation

---

## 8. Submission Requirements

### 📦 What Will You Submit?

⚠️ **IMPORTANT:** This section contains visual submission requirements displayed in the presentation.

**Action Required:** Please refer to Slide 7 in the original PowerPoint to view the complete submission requirements and folder structure.

**Expected Components:**

- Project folder with specific naming convention
- Agent files in correct directory structure
- JSON output files
- (Full details visible in slide image)

---

## 9. Restrictions

### 🛑 Competition Restrictions

⚠️ **IMPORTANT:** This section contains visual restrictions displayed in the presentation.

**Action Required:** Please refer to Slide 8 in the original PowerPoint to view the complete list of restrictions.

**Key Areas Likely Covered:**

- Model usage restrictions
- API access limitations
- External resource constraints
- Training data limitations
- (Full details visible in slide image)

---

## 10. Token & Time Limits

### ⏱️ Performance Requirements

This section contains **CRITICAL** limits that your agents must adhere to:

### Token Limits

#### **For Q-Agent (Question Agent):**

**Maximum Token Allocation:**

- **150 tokens cumulatively** for the content corresponding to:
  - `topic`
  - `question`
  - `choices`
  - `answer`

**Important Notes:**

- This **excludes** token count for:
  - Double quotes (`"`)
  - String length for `topic`, `question`, `choices`, and `answer` string keys

**Explanation Tokens:**

- Remaining tokens for `explanation`: **1024 - 150 = 874 tokens**
- Must be generated **within the time limit**

#### **For A-Agent (Answer Agent):**

- Token limits to be verified (refer to sample_answer.json)

### Time Limits

#### **Q-Agent Time Limits:**

- **Per Question:** Each question must be generated in **under 13 seconds**
- **Total for 100 Questions:** No more than **1300 seconds** (~21 minutes)

#### **A-Agent Time Limits:**

- **Per Answer:** Each answer must be generated in **under 9 seconds**
- **Total for 100 Answers:** No more than **900 seconds** (~15 minutes)

### ⚠️ CRITICAL NOTE:
**"We will only consider those questions' and answers' JSON files that remain under the time limit."**

This means:

- If your Q-Agent takes longer than 1300 seconds for 100 questions, your submission will be **disqualified**
- If your A-Agent takes longer than 900 seconds for 100 answers, your submission will be **disqualified**
- Individual questions/answers exceeding their time limits may be **skipped or penalized**

---

## 11. JSON Format Specifications

### 📄 Q-Agent Output Format

The Q-Agent must generate a JSON object with the following structure:

```json
{
  "topic": "<Topic of the Question>",
  "question": "<full question text>",
  "choices": [
    "A) <choice A text>",
    "B) <choice B text>",
    "C) <choice C text>",
    "D) <choice D text>"
  ],
  "answer": "<correct choice letter only>",
  "explanation": "<brief explanation within 100 words for why the answer is correct>"
}
```

### Field Specifications:

#### 1. **topic** (String)

- Must be one of the four approved topics:
  - "Syllogisms"
  - "Seating Arrangements"
  - "Blood Relations and Family Tree"
  - "Mixed Series (Alphanumeric)"

#### 2. **question** (String)

- The complete question text
- Should be clear and unambiguous
- Must relate to the specified topic

#### 3. **choices** (Array of Strings)

- Exactly **4 choices** (A, B, C, D)
- Each choice must start with the letter followed by ")"
- Format: `"A) <choice text>"`
- All choices should be plausible

#### 4. **answer** (String)

- **Must contain ONLY the correct choice letter**
- Valid values: `"A"`, `"B"`, `"C"`, or `"D"`
- No parentheses, no additional text

#### 5. **explanation** (String)

- Brief explanation for why the answer is correct
- **Maximum: 100 words**
- Should be clear and logical
- Must fit within the 874 token allocation

### Token Count Requirements:

- Combined tokens for `topic`, `question`, `choices`, `answer`: **≤ 150 tokens**
- Tokens for `explanation`: **≤ 874 tokens** (within time limit)
- **Total: ≤ 1024 tokens**

### 📄 A-Agent Input/Output Format

**Note:** The A-Agent format specifications should be detailed in `sample_answer.json`.

**Expected Input:**

- The A-Agent receives the question JSON from the Q-Agent
- Must parse and understand the question structure

**Expected Output:**

- The A-Agent must provide its answer in the required format
- Format details available in `sample_answer.json`
- Likely contains:
  - Selected answer (A/B/C/D)
  - Reasoning/explanation (optional)

**Action Required:** Refer to `sample_answer.json` for complete A-Agent format specifications.

---

## 12. Takeaways

### 🎯 Key Takeaways

⚠️ **IMPORTANT:** This section contains visual takeaways displayed in the presentation.

**Action Required:** Please refer to Slide 10 in the original PowerPoint to view the complete takeaways.

**Expected Content:**

- Critical reminders
- Best practices
- Common pitfalls to avoid
- Success strategies
- (Full details visible in slide image)

---

## 13. File Structure & Naming Conventions

### 📁 Mandatory Naming Conventions

**⚠️ STRICT ADHERENCE REQUIRED**

#### Root Folder Configuration:

**Folder Name Format:**
```
AAIPL_your_IP
```

**Requirements:**

- `your_IP` must be your **IPv4 address**
- Replace dots (`.`) with **underscores** (`_`)
- **No special characters allowed**

**Example:**

- If your IP is `192.168.1.100`
- Folder name: `AAIPL_192_168_1_100`

### Agent File Structure

You must submit **FOUR** specific Python files in the `agents/` directory:

#### **For Q-Agent:**

1. **Wrapper File:** `agents/question_agent.py`
2. **Model File:** `agents/question_model.py`

#### **For A-Agent:**

1. **Wrapper File:** `agents/answer_agent.py`
2. **Model File:** `agents/answer_model.py`

**File Structure:**
```
AAIPL_your_IP/
├── agents/
│   ├── question_agent.py    # Q-Agent wrapper
│   ├── question_model.py    # Q-Agent model
│   ├── answer_agent.py      # A-Agent wrapper
│   └── answer_model.py      # A-Agent model
├── questions.json           # Generated questions (output)
├── answers.json             # Generated answers (output)
└── [other supporting files]
```

---

## 14. Additional Information

### Final Slide
🎉 **All the BEST!** 🎉

---

## Summary Checklist

Before submission, ensure you have:

- [ ] Used a WhiteList model for both Q-Agent and A-Agent
- [ ] Named your root folder correctly (AAIPL_your_IP)
- [ ] Created all four required Python files in `agents/` directory
- [ ] Generated questions covering all four topics
- [ ] Avoided numeric-style seating arrangement questions
- [ ] Ensured Q-Agent generates questions within 13 seconds each
- [ ] Ensured A-Agent generates answers within 9 seconds each
- [ ] Total Q-Agent time ≤ 1300 seconds for 100 questions
- [ ] Total A-Agent time ≤ 900 seconds for 100 answers
- [ ] Q-Agent JSON follows exact format with token limits
- [ ] A-Agent JSON follows format specified in sample_answer.json
- [ ] Reviewed Guidelines (Slide 4)
- [ ] Reviewed Submission Requirements (Slide 7)
- [ ] Reviewed Restrictions (Slide 8)
- [ ] Reviewed Takeaways (Slide 10)
- [ ] Tested your agents with sample files
- [ ] Verified all outputs are valid JSON

---

## Important Notes

⚠️ **Slides Requiring Manual Review:**

The following slides contain important visual information that must be reviewed in the original PowerPoint presentation:

1. **Slide 4 - Guidelines:** Contains visual guidelines and rules
2. **Slide 7 - Submission Requirements:** Shows complete submission structure
3. **Slide 8 - Restrictions:** Lists all competition restrictions
4. **Slide 10 - Takeaways:** Key points and reminders

**Action:** Please open the original PowerPoint file and carefully review these slides to ensure you have complete information.

---

## Document Information

- **Source:** AAIPL_Overview_IITD.pptx
- **Total Slides:** 11
- **Institution:** IIT Delhi
- **Competition:** AMD AI Premier League (AAIPL)

---

**End of Document**
project_rules.md
16 KB