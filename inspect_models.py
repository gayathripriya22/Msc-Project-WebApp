import pickle
import os

files = ['label_encoder.pkl', 'onehot_encoder.pkl', 'random_forest_model.pkl', 'xgboost_model.pkl']
for name in files:
    path = os.path.join(os.getcwd(), name)
    print('===', name, '===')
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(type(obj))
        print('classes_', getattr(obj, 'classes_', None))
        print('feature_names_in_', getattr(obj, 'feature_names_in_', None))
        print('params keys', list(obj.get_params().keys())[:20] if hasattr(obj, 'get_params') else None)
    except Exception as e:
        print('ERROR', repr(e))
