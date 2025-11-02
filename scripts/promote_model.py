# promote model

import os
import json
import logging

logging.basicConfig(level=logging.INFO)


def load_registry():
    """Load the model registry."""
    registry_path = 'reports/model_registry.json'
    if not os.path.exists(registry_path):
        logging.error('Model registry not found')
        return None
    
    with open(registry_path, 'r') as f:
        return json.load(f)


def save_registry(registry):
    """Save the model registry."""
    with open('reports/model_registry.json', 'w') as f:
        json.dump(registry, f, indent=4)


def get_model_score(version_data):
    """Calculate a composite score for model comparison."""
    metrics = version_data.get('metrics', {})
    
    # Use accuracy as primary metric, or weighted composite
    accuracy = metrics.get('accuracy', 0)
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    f1 = metrics.get('f1_score', 0)
    auc = metrics.get('auc', 0)
    
    # Weighted score: prioritize accuracy and AUC
    score = (accuracy * 0.4 + auc * 0.3 + f1 * 0.2 +
             precision * 0.05 + recall * 0.05)
    
    return score


def promote_best_model():
    """Promote the best staging model to production."""
    registry = load_registry()
    if not registry:
        return
    
    model_name = "my_model"
    if model_name not in registry.get("models", {}):
        logging.error(f'Model {model_name} not found in registry')
        return
    
    model_data = registry["models"][model_name]
    versions = model_data.get("versions", [])
    
    if not versions:
        logging.error('No model versions found')
        return
    
    # Find all staging models
    staging_versions = [
        v for v in versions if v.get('status') == 'staging'
    ]
    
    if not staging_versions:
        logging.info('No staging models to promote')
        return
    
    # Find the best staging model
    best_version = max(staging_versions, key=get_model_score)
    best_score = get_model_score(best_version)
    
    logging.info(
        f'Best staging model: v{best_version["version"]} '
        f'(score: {best_score:.4f})'
    )
    
    # Archive all current production models
    for version in versions:
        if version.get('status') == 'production':
            version['status'] = 'archived'
            logging.info(f'Archived v{version["version"]}')
    
    # Archive lower-performing staging models
    for version in staging_versions:
        if version['version'] != best_version['version']:
            version['status'] = 'archived'
            logging.info(f'Archived v{version["version"]}')
    
    # Promote best model to production
    best_version['status'] = 'production'
    logging.info(
        f'Promoted v{best_version["version"]} to production'
    )
    
    # Save updated registry
    save_registry(registry)
    
    # Print summary
    print('\n' + '='*80)
    print('🎉 MODEL PROMOTION COMPLETED')
    print('='*80)
    print(f'📦 Model: {model_name}')
    print(f'🔢 Promoted Version: v{best_version["version"]}')
    print(f'📊 Score: {best_score:.4f}')
    print(f'\n📈 Metrics:')
    for k, v in best_version.get('metrics', {}).items():
        print(f'   • {k}: {v:.4f}')
    print('='*80 + '\n')


if __name__ == "__main__":
    promote_best_model()
