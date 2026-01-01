# Question Annotation Tool

A web-based tool for manually annotating forecasting questions as good or bad.

## Features

- **Visual Question Review**: Display question, background, resolution criteria, answer type, and answer
- **Article Context**: View the source article content (collapsible)
- **Simple Annotation**: Click "YES" (good) or "NO" (bad) buttons, or use keyboard shortcuts (Y/N)
- **Progress Tracking**: See statistics on total, annotated, approved, and rejected questions
- **Filtering**: Filter questions by status (all, pending, approved, rejected)
- **Auto-save**: Annotations are automatically saved to the JSONL file with backups
- **Keyboard Navigation**: Use arrow keys to navigate between questions

## Usage

### Basic Usage

```bash
cd qgen/adversarial
python annotator.py --data_path /path/to/your/questions.jsonl --port 5001
```

### Example with the test file

```bash
python annotator.py --data_path /fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_manualFilter.jsonl --port 5001
```

Then open your browser to: `http://localhost:5001`

### Command Line Arguments

- `--data_path`: Path to the JSONL file containing questions (required)
- `--port`: Port to run the server on (default: 5001)

## How It Works

1. **Load Questions**: The tool loads all questions from the specified JSONL file
2. **Browse and Annotate**: Navigate through questions and mark them as good (1) or bad (0)
3. **Auto-save**: Each annotation adds a `human_filter` field to the question:
   - `human_filter: 1` = Good question (approved)
   - `human_filter: 0` = Bad question (rejected)
4. **Backup**: Before saving, a timestamped backup is created automatically

## Data Format

### Input
Each question in the JSONL file should have these fields:
- `question_title`: The main question text
- `background`: Background information
- `resolution_criteria`: How the question will be resolved
- `answer_type`: Type of answer expected
- `answer`: The correct answer
- `article_maintext`: Source article content
- `article_title`, `article_description`, etc.: Article metadata

### Output
The tool adds the `human_filter` field to each annotated question:
```json
{
  "qid": "1",
  "question_title": "...",
  "background": "...",
  "human_filter": 1  // 1 for approved, 0 for rejected
}
```

## Keyboard Shortcuts

- `Y`: Mark question as good (yes)
- `N`: Mark question as bad (no)
- `←`: Previous question
- `→`: Next question

## Safety Features

- **Automatic Backups**: Creates timestamped backups before each save
- **Backup Format**: `filename.jsonl.backup_YYYYMMDD_HHMMSS`
- **Overwrites**: Each annotation overwrites the previous value if you change your mind

## Tips

1. Start from the beginning and work through systematically
2. Use keyboard shortcuts for faster annotation
3. The tool auto-advances to the next question after annotation
4. Check the progress bar to track your completion
5. Use filters to review your annotations

