import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import os

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a dataset class specifically for financial text
class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):  # Increased max_len for financial texts
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_balanced_dataset():
    """Load and prepare the balanced financial sentiment dataset"""
    print("Loading the combined financial sentiment dataset...")

    # Load the balanced dataset you created (with equal representation of sentiment classes)
    train_df = pd.read_csv('/content/finbert_train_data .csv')
    val_df = pd.read_csv('/content/finbert_val_data .csv')
    test_df = pd.read_csv('/content/finbert_test_data .csv')

    print(f"Dataset loaded: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")

    # Map sentiment labels to match ProsusAI/finbert format
    sentiment_map = {
        'positive': 0,
        'negative': 1,
        'neutral': 2
    }

    # Apply mapping and prepare data
    train_texts = train_df['text'].tolist()
    val_texts = val_df['text'].tolist()
    test_texts = test_df['text'].tolist()

    train_labels = [sentiment_map[label.lower()] for label in train_df['sentiment']]
    val_labels = [sentiment_map[label.lower()] for label in val_df['sentiment']]
    test_labels = [sentiment_map[label.lower()] for label in test_df['sentiment']]

    # Display class distribution
    print("Class distribution:")
    print(f"Train: {pd.Series(train_labels).value_counts().sort_index().to_dict()}")
    print(f"Validation: {pd.Series(val_labels).value_counts().sort_index().to_dict()}")
    print(f"Test: {pd.Series(test_labels).value_counts().sort_index().to_dict()}")

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def train_finbert_model(train_data, val_data):
    """Train the FinBERT model on financial sentiment data"""
    print("Initializing ProsusAI/finbert model...")
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
    model.to(device)

    train_texts, train_labels = train_data
    val_texts, val_labels = val_data

    # Create datasets and dataloaders
    train_dataset = FinancialSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = FinancialSentimentDataset(val_texts, val_labels, tokenizer)

    batch_size = 16
    if device.type == 'cpu':
        batch_size = 8  # Smaller batch size for CPU

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    total_steps = len(train_loader) * 4  # 4 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )

    # Training loop
    best_accuracy = 0
    training_stats = []

    print("Starting training...")
    for epoch in range(4):
        print(f"{'='*20} Epoch {epoch+1}/4 {'='*20}")

        # Training phase
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch+1}")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                val_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(true_labels, predictions)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} results:")
        print(f"  Training loss: {avg_train_loss:.4f}")
        print(f"  Validation loss: {avg_val_loss:.4f}")
        print(f"  Validation accuracy: {val_accuracy:.4f}")

        # Save statistics
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        })

        # Save best model
        if val_accuracy > best_accuracy:
            print(f"Validation accuracy improved from {best_accuracy:.4f} to {val_accuracy:.4f}")
            best_accuracy = val_accuracy

            # Save model
            print("Saving model...")
            model_path = 'finbert_best_model'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

    # Plot training progress
    stats_df = pd.DataFrame(training_stats)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(stats_df['epoch'], stats_df['train_loss'], label='Training Loss')
    plt.plot(stats_df['epoch'], stats_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(stats_df['epoch'], stats_df['val_accuracy'], 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig('finbert_training_metrics.png')
    plt.close()

    print(f"Training complete. Best validation accuracy: {best_accuracy:.4f}")
    return model, tokenizer, best_accuracy

def evaluate_finbert_model(model, tokenizer, test_data):
    """Evaluate the FinBERT model on test data"""
    test_texts, test_labels = test_data

    # Create test dataset
    test_dataset = FinancialSentimentDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Evaluation
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(
        true_labels,
        predictions,
        target_names=['Positive', 'Negative', 'Neutral'],
        digits=4
    )

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Positive', 'Negative', 'Neutral'],
        yticklabels=['Positive', 'Negative', 'Neutral']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('finbert_confusion_matrix.png')
    plt.close()

    print("=== Test Results ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    return accuracy, report

def test_sentiment_analysis(model, tokenizer):
    """Test the model on specific financial texts"""
    example_texts = [
        "The company reported strong earnings, beating analyst expectations.",
        "The stock price has been falling steadily over the past week.",
        "The market remained largely unchanged today with minimal fluctuations.",
        "The company announced layoffs, causing investor concerns.",
        "The new product launch is expected to boost quarterly revenues."
    ]

    label_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
    model.eval()

    results = []
    for text in example_texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        sentiment_id = torch.argmax(probabilities).item()
        sentiment = label_map[sentiment_id]
        confidence = probabilities[sentiment_id].item()

        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'positive': probabilities[0].item(),
                'negative': probabilities[1].item(),
                'neutral': probabilities[2].item()
            }
        })

    print("\n=== Sample Sentiment Analysis Results ===")
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Predicted sentiment: {result['sentiment']} (confidence: {result['confidence']:.4f})")
        print(f"Probabilities: Positive={result['probabilities']['positive']:.4f}, " +
              f"Negative={result['probabilities']['negative']:.4f}, " +
              f"Neutral={result['probabilities']['neutral']:.4f}")

    return results

# Main execution
def main():
    print("Starting FinBERT sentiment analysis model training and evaluation")

    # Load the balanced dataset
    train_data, val_data, test_data = load_balanced_dataset()

    # Train the model
    model, tokenizer, best_accuracy = train_finbert_model(train_data, val_data)

    # Evaluate on test data
    test_accuracy, report = evaluate_finbert_model(model, tokenizer, test_data)

    # Test on example sentences
    sentiment_results = test_sentiment_analysis(model, tokenizer)

    print("\nFinBERT model training and evaluation complete!")
    print(f"Final model accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
