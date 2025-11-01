import json
import os

def view_models():
    """View all registered models from the local registry."""
    registry_path = 'reports/model_registry.json'
    
    if not os.path.exists(registry_path):
        print("❌ No model registry found!")
        print(f"   Looking for: {registry_path}")
        return
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    if "models" not in registry or not registry["models"]:
        print("❌ No models registered yet.")
        return
    
    print(f'\n{"="*80}')
    print(f'📚 YOUR REGISTERED MODELS')
    print(f'{"="*80}\n')
    
    for model_name, model_data in registry["models"].items():
        print(f'🤖 Model: {model_name}')
        print(f'📝 Description: {model_data.get("description", "N/A")}')
        print(f'📅 Created: {model_data.get("created_at", "N/A")}')
        print(f'🔄 Last Updated: {model_data.get("updated_at", "N/A")}')
        print(f'🏷️  Latest Version: v{model_data.get("latest_version", 1)}')
        print(f'📊 Total Versions: {len(model_data.get("versions", []))}')
        print(f'\n{"─"*80}')
        print(f'VERSION HISTORY:')
        print(f'{"─"*80}\n')
        
        for version in reversed(model_data.get("versions", [])):  # Show latest first
            print(f'  🏷️  Version {version["version"]} - {version.get("status", "unknown").upper()}')
            print(f'  📅 Registered: {version.get("registered_at", "N/A")}')
            print(f'  🆔 Run ID: {version.get("run_id", "N/A")}')
            print(f'  📁 Model File: {version.get("model_path", "N/A")}')
            
            if version.get("metrics"):
                print(f'  📈 Metrics:')
                for k, v in version.get("metrics", {}).items():
                    print(f'     • {k}: {v:.4f}')
            
            print(f'  🔗 Links:')
            print(f'     • Run: {version.get("mlflow_run_url", "N/A")}')
            print(f'     • Experiment: {version.get("mlflow_experiment_url", "N/A")}')
            print()
        
        print(f'{"-"*80}\n')
    
    print(f'💡 Tip: The "Models" tab in DagHub UI is empty because it requires a paid plan.')
    print(f'   Your models are safely stored locally and tracked in MLflow experiments!')
    print(f'{"="*80}\n')

if __name__ == '__main__':
    view_models()