import json
import os
import argparse
from flask import Flask, render_template, request, jsonify
import shutil
from datetime import datetime

def create_app(data_path):
    # Create the Flask app with the correct template folder path
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
    
    # Extract model name from data path for display purposes
    model_name = os.path.basename(data_path).replace('.jsonl', '')
    
    def load_data():
        try:
            with open(data_path, 'r') as f:
                data = [json.loads(line) for line in f if line.strip()]
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def save_data(data):
        """Save data back to the JSONL file with backup"""
        try:
            # Create backup
            backup_path = data_path + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            shutil.copy2(data_path, backup_path)
            print(f"Created backup at: {backup_path}")
            
            # Write updated data
            with open(data_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

    @app.route('/')
    def index():
        data = load_data()
        # Count annotations
        total = len(data)
        annotated = sum(1 for item in data if 'human_filter' in item)
        approved = sum(1 for item in data if item.get('human_filter') == 1)
        rejected = sum(1 for item in data if item.get('human_filter') == 0)
        
        return render_template('annotation_index.html', 
                             articles=data, 
                             model_name=model_name,
                             total=total,
                             annotated=annotated,
                             approved=approved,
                             rejected=rejected)

    @app.route('/annotate/<int:question_id>')
    def annotate(question_id):
        data = load_data()
        if 0 <= question_id < len(data):
            return render_template('annotation_detail.html', 
                                 question=data[question_id], 
                                 question_id=question_id,
                                 total=len(data),
                                 model_name=model_name)
        return "Question not found", 404
    
    @app.route('/api/annotate/<int:question_id>', methods=['POST'])
    def save_annotation(question_id):
        """Save annotation for a specific question"""
        try:
            annotation = request.json.get('annotation')  # 1 for yes, 0 for no
            
            if annotation not in [0, 1]:
                return jsonify({'success': False, 'error': 'Invalid annotation value'}), 400
            
            data = load_data()
            
            if 0 <= question_id < len(data):
                data[question_id]['human_filter'] = annotation
                
                if save_data(data):
                    return jsonify({'success': True})
                else:
                    return jsonify({'success': False, 'error': 'Failed to save data'}), 500
            else:
                return jsonify({'success': False, 'error': 'Question not found'}), 404
                
        except Exception as e:
            print(f"Error saving annotation: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/questions')
    def get_questions():
        """API endpoint for paginated questions"""
        data = load_data()
        page = int(request.args.get('page', 0))
        per_page = int(request.args.get('per_page', 20))
        
        start = page * per_page
        end = start + per_page
        
        questions = data[start:end]
        return jsonify({
            'questions': questions,
            'total': len(data),
            'page': page,
            'per_page': per_page
        })
    
    return app

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Annotate questions for quality')
    parser.add_argument('--data_path', type=str, 
                        default="/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_manualFilter.jsonl",
                        help='Path to the JSONL file with the questions')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
    
    args = parser.parse_args()
    
    # Create and run the app
    app = create_app(args.data_path)
    app.run(host='0.0.0.0', port=args.port, debug=True)

