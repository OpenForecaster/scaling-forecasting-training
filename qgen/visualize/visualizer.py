import json
import os
import argparse
from flask import Flask, render_template, request, jsonify
import os.path

def create_app(data_path, is_freeq=False):
    # Create the Flask app with the correct template folder path
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
    
    # Extract model name from data path for display purposes
    model_name = os.path.basename(data_path).split('_')[-1] if '_' in os.path.basename(data_path) else os.path.basename(data_path)
    
    def load_data():
        try:
            with open(data_path, 'r') as f:
                # Handle both JSON and JSONL formats
                if data_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    @app.route('/')
    def index():
        data = load_data()
        print(os.getcwd())
        return render_template('index.html', articles=data, model_name=model_name, is_freeq=is_freeq)

    @app.route('/article/<int:article_id>')
    def article_detail(article_id):
        data = load_data()
        if 0 <= article_id < len(data):
            return render_template('article_detail.html', article=data[article_id], article_id=article_id, model_name=model_name, is_freeq=is_freeq)
        return "Article not found", 404

    @app.route('/api/articles')
    def get_articles():
        data = load_data()
        page = int(request.args.get('page', 0))
        per_page = int(request.args.get('per_page', 10))
        
        start = page * per_page
        end = start + per_page
        
        articles = data[start:end]
        return jsonify({
            'articles': articles,
            'total': len(data),
            'page': page,
            'per_page': per_page
        })
    
    return app

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize model outputs')
    parser.add_argument('--data_path', type=str, 
                        default="/is/cluster/fast/sgoel/forecasting/qgen/reuters/selected1000_2021-2022_deepseekv3024",
                        help='Path to the JSON or JSONL file with the data')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--freeq', action='store_true', help='Visualize free-form questions instead of MCQs')
    
    args = parser.parse_args()
    
    # Create and run the app
    app = create_app(args.data_path, args.freeq)
    app.run(host='0.0.0.0', port=args.port, debug=True)