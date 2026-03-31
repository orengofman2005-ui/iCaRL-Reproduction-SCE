# iCaRL: Incremental Classifier and Representation Learning - Reproduction Project

## Project Overview

This repository contains a full reproduction of the paper ["iCaRL: Incremental Classifier and Representation Learning" (Rebuffi et al., CVPR 2017)](https://arxiv.org/pdf/1611.07725). The focus of this implementation is to evaluate the model's ability to handle Class-Incremental Learning (CIL) and mitigate the effects of Catastrophic Forgetting.

## Motivation: Why iCaRL?

**למה בחרתי בפרויקט הזה? (מנקודת המבט שלי כסטודנט שנה ג' להנדסת חשמל)**
כסטודנט להנדסת חשמל, תחום הלמידה העמוקה (Deep Learning) ועיבוד נתונים מרתק אותי, במיוחד כשחושבים על החיבור שבין אלגוריתמיקה למערכות חכמות בעולם האמיתי (כמו מערכות משובצות מחשב, רובוטיקה או רכבים אוטונומיים). אחת הבעיות המרכזיות בהטמעת בינה מלאכותית בשטח היא "שכחה קטסטרופלית" (Catastrophic Forgetting) – מודלים מתקשים ללמוד מידע חדש באופן רציף מבלי לדרוס ולאבד את הידע הקודם שלהם. אימון מחדש מאפס דורש משאבי חישוב, אנרגיה וזיכרון עצומים שלרוב אינם זמינים במערכות קצה.

המאמר של iCaRL מציג גישה פרקטית ואלגנטית שמשלבת ניהול זיכרון חכם (Exemplars) ולמידת ייצוגים יציבה, ומאפשרת למודל להמשיך ללמוד באופן אינקרמנטלי עם מגבלות משאבים. עבורי, השחזור של המאמר הזה הוא הזדמנות מעולה לצלול לקרביים של ארכיטקטורות ResNet, להבין לעומק איך מנהלים זיכרון וחישוב בצורה אופטימלית ברשתות נוירונים, ולהתמודד עם כתיבת קוד ותהליכי אימון מורכבים – סט כלים שהוא הכרחי לכל מהנדס בתחומי ה-AI, האלגוריתמיקה ועיבוד האותות.

## Technical Methodology

*   **Dataset**: CIFAR-100, consisting of 100 distinct object classes.
*   **Incremental Protocol**: The dataset is partitioned into 10 sequential tasks, with 10 new classes introduced in each phase.
*   **Architecture**: Implementation based on the PyCIL framework (Python Class-Incremental Learning) using a ResNet32 backbone.
*   **Optimization**: Training is performed on a local NVIDIA GPU with specific CUDA device configurations.

## Technical Troubleshooting & AI Collaboration

The development and environment setup involved significant technical problem-solving, documented in the `/AI_Documentation` folder. Key issues addressed include:

*   **Runtime Optimization**: Resolving OpenMP and `libiomp5md.dll` initialization conflicts.
*   **Hardware Interfacing**: Correcting GPU Device ID assertions for single-GPU environments.
*   **Hyperparameter Tuning**: Aligning JSON configuration files with the original paper’s benchmarks (Seed 1993, Memory Size 2000).

## Execution Guide

1.  Install the required environment:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the training process:
    ```bash
    python main.py --config=./exps/icarl.json
    ```

## Experimental Results

Quantitative results, including Top-1 Accuracy across incremental steps and comparison graphs with the original paper, will be populated here upon execution completion.