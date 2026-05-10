import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.cnn import CNNClassifier
from data_utils.dataset import CompressorClassifierDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_cm(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def get_ensemble_predictions(model_files, loader):
    models = []
    for f in model_files:
        m = CNNClassifier(n_attack_classes=6, n_release_classes=4).to(DEVICE)
        m.load_state_dict(torch.load(f, map_location=DEVICE))
        m.eval()
        models.append(m)

    all_attack_probs, all_release_probs, all_targets = [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            batch_a_probs, batch_r_probs = [], []
            
            for m in models:
                out_a, out_r = m(inputs)
                batch_a_probs.append(torch.softmax(out_a, dim=1))
                batch_r_probs.append(torch.softmax(out_r, dim=1))
            
            avg_a = torch.stack(batch_a_probs).mean(dim=0)
            avg_r = torch.stack(batch_r_probs).mean(dim=0)
            
            all_attack_probs.append(avg_a.cpu().numpy())
            all_release_probs.append(avg_r.cpu().numpy())
            all_targets.append(targets.numpy())

    return np.vstack(all_attack_probs), np.vstack(all_release_probs), np.vstack(all_targets)

def main():
    model_files = [f'best_classifier_cnn_fold_{i+1}.pth' for i in range(5)]
    # Importante: Usamos stage='all' o una partición de test específica si la tienes
    test_dataset = CompressorClassifierDataset('data_ready/metadata.csv', 'data_ready', stage='all')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    probs_a, probs_r, targets = get_ensemble_predictions(model_files, test_loader)
    preds_a, preds_r = np.argmax(probs_a, axis=1), np.argmax(probs_r, axis=1)

    # Cálculo de métricas
    acc_a = accuracy_score(targets[:, 0], preds_a)
    acc_r = accuracy_score(targets[:, 1], preds_r)

    # Guardar porcentajes en CSV
    results_df = pd.DataFrame({
        'Metric': ['Accuracy Attack', 'Accuracy Release'],
        'Value': [acc_a, acc_r]
    })
    results_df.to_csv('ensemble_results.csv', index=False)

    # Etiquetas (ajusta según tus valores reales de metadata.csv)
    attack_labels = ['0.1', '0.3', '1.0', '3.0', '10.0', '30.0']
    release_labels = ['0.1', '0.2', '0.6', '1.2']

    # Generar y guardar matrices
    plot_cm(targets[:, 0], preds_a, attack_labels, f'Ensemble Attack CM (Acc: {acc_a:.2f})', 'ensemble_cm_attack.png')
    plot_cm(targets[:, 1], preds_r, release_labels, f'Ensemble Release CM (Acc: {acc_r:.2f})', 'ensemble_cm_release.png')

    print(f"\n✅ Resultados del Ensemble guardados.")
    print(f"Attack Acc: {acc_a:.4f} | Release Acc: {acc_r:.4f}")

if __name__ == "__main__":
    main()