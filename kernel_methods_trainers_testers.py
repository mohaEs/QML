from kernel_methods_utils import load_and_shuffle_data
import pickle 
import os

def run(model_func, dataset_name, num_qubits=4, seed=42, output_dir=data_dir):
    model_name = model_func.__name__.replace("run_", "")
    is_quantum = model_name.startswith("q")
    print(f"Running {model_name.upper()} experiment on {dataset_name} dataset")

    print(f"Loading and preprocessing data with seed {seed}...")
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_and_shuffle_data(
        seed=seed, num_qubits=num_qubits)

    print(f"Training and evaluating {model_name.upper()}...")
    if is_quantum:
        model_result = model_func(X_train, y_train, X_val, y_val, X_test, y_test, num_qubits=num_qubits)
    else:
        model_result = model_func(X_train, y_train, X_val, y_val, X_test, y_test)

    result = {
        'model': model_name,
        'dataset': dataset_name,
        'seed': seed,
        'class_names': class_names,
        'y_test': y_test,
        'is_quantum': is_quantum,
        'num_qubits': num_qubits if is_quantum else None,
        **model_result
    }

    print(f"\n{model_name.upper()} Results:")
    print("  Validation Metrics:")
    print(f"    Accuracy:  {model_result['val_accuracy']:.4f}")
    print(f"    Precision: {model_result['val_precision']:.4f}")
    print(f"    Recall:    {model_result['val_recall']:.4f}")
    print(f"    F1 Score:  {model_result['val_f1']:.4f}")
    print("  Test Metrics:")
    print(f"    Accuracy:  {model_result['accuracy']:.4f}")
    print(f"    Precision: {model_result['precision']:.4f}")
    print(f"    Recall:    {model_result['recall']:.4f}")
    print(f"    F1 Score:  {model_result['f1']:.4f}")

    output_file = os.path.join(output_dir, f'{model_name}_{dataset_name}_results.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"Results saved to {output_file}")

    return result
