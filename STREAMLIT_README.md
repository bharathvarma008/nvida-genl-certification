# NCP-GENL Study Dashboard - Streamlit App

## Overview
Interactive Streamlit dashboard for tracking your NCP-GENL certification preparation progress, managing mock exams, and reviewing flashcards.

## Features

### 📊 Progress Dashboard
- **Overall Progress Tracking**: Study hours, practice questions, days remaining
- **Domain Mastery Visualization**: Visual charts showing your progress across all 10 domains
- **Practice Question Statistics**: Track questions by domain with accuracy metrics
- **Daily Study Hours Log**: Log and visualize your daily study sessions
- **Mock Exam Results Summary**: Quick overview of your mock exam performance
- **Success Metrics**: Track key metrics to ensure you're on target

### 📝 Mock Exams
- **Mock Exam 1 & 2 Management**: Enter and track results for both mock exams
- **Domain Breakdown Analysis**: See your performance by domain
- **Weak Area Identification**: Automatically identifies domains with <70% accuracy
- **Comparison View**: Compare Mock Exam 1 vs Mock Exam 2 to track improvement
- **Visual Charts**: Interactive charts showing your progress

### 🃏 Flashcards
- **Interactive Flashcard System**: Review flashcards by domain
- **Status Tracking**: Mark cards as New, Need Review, or Mastered
- **Spaced Repetition**: Track review counts and last reviewed dates
- **Domain Filtering**: Focus on specific domains or review all
- **Progress Tracking**: See statistics on your flashcard mastery

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

## Running the App

1. **Start the Streamlit app**:
```bash
streamlit run app.py
```

2. **Open your browser**: The app will automatically open at `http://localhost:8501`

## First Time Setup

1. **Set Study Dates**: 
   - When you first open the app, you'll be prompted to set your study start date and target exam date
   - These dates are used to calculate days remaining and track your progress

2. **Start Logging**:
   - Add your first study hours in the Progress Dashboard
   - Log practice questions as you complete them
   - Enter mock exam results when you take them
   - Review flashcards daily

## Data Storage

- All data is stored locally in `study_data.json`
- Your data persists between sessions
- You can backup this file to save your progress

## Usage Tips

### Progress Dashboard
- **Log Study Hours Daily**: Use the "Add Study Hours" section to track your daily study time
- **Update Practice Questions**: Manually update practice question counts (or integrate with your practice tracker)
- **Monitor Domain Mastery**: Keep an eye on domains below 75% accuracy

### Mock Exams
- **Enter Results Immediately**: After taking a mock exam, enter results while they're fresh
- **Review Domain Breakdown**: Use the domain breakdown to identify weak areas
- **Track Improvement**: Compare Mock 1 vs Mock 2 to see your progress

### Flashcards
- **Daily Review**: Review 20-30 flashcards daily during evening sessions
- **Focus on Weak Areas**: Prioritize domains you're struggling with
- **Mark Status Accurately**: Be honest about whether you've mastered a concept
- **Use Spaced Repetition**: Review "Need Review" cards more frequently

## Features in Detail

### Progress Tracking
- **Study Hours**: Target 50+ hours over 14 days (3.5-5h/day)
- **Practice Questions**: Target 400+ questions
- **Domain Mastery**: Target >75% accuracy in all domains
- **Visual Indicators**: Color-coded status (⬜ Not Started, 🟡 In Progress, 🟢 On Target, ✅ Mastered)

### Mock Exam Analysis
- **Score Tracking**: Overall score and domain-specific scores
- **Time Management**: Track time taken vs 120-minute target
- **Weak Area Detection**: Automatically highlights domains <70% accuracy
- **Improvement Tracking**: Compare performance between mock exams

### Flashcard System
- **100+ Topics**: Covers all 10 domains with key concepts
- **Status Management**: New → Need Review → Mastered progression
- **Review Tracking**: Counts reviews and tracks last reviewed date
- **Domain Organization**: Organized by exam domains for easy navigation

## Troubleshooting

### App won't start
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ recommended)

### Data not saving
- Check file permissions in the directory
- Ensure `study_data.json` is writable

### Charts not displaying
- Make sure plotly is installed: `pip install plotly`
- Try refreshing the browser

## Backup Your Data

To backup your progress:
```bash
cp study_data.json study_data_backup.json
```

To restore:
```bash
cp study_data_backup.json study_data.json
```

## Next Steps

1. **Start Using**: Run the app and set your study dates
2. **Daily Logging**: Make it a habit to log study hours and practice questions
3. **Mock Exams**: Enter results immediately after taking mock exams
4. **Flashcard Review**: Review flashcards daily, especially weak areas
5. **Track Progress**: Use the dashboard to ensure you're meeting your targets

## Support

For issues or questions:
- Check the main README.md for study plan details
- Review the markdown files for detailed information
- Ensure you're following the 14-day intensive study plan

---

**Happy Studying! 🎓**
