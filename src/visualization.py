"""
Visualization utilities for Comment-Sentiment prediction explanations.

This module provides functions to visualize model predictions, token importances, and explanations using LIME or similar libraries.
Plots are saved to the /visualizations directory.
"""
import os
import matplotlib.pyplot as plt

# Optional: LIME for transformers
try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None

def _get_bar_colors(importances, pos_color, neg_color):
    return [neg_color if s < 0 else pos_color for s in importances]

def _draw_bar_labels(ax, bars, fontsize):
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (0.004 if width >= 0 else -0.004),
            bar.get_y() + bar.get_height() / 2,
            f'{width:.2f}',
            va='center',
            ha='left' if width >= 0 else 'right',
            fontsize=fontsize,
            color='#333',
            fontweight='normal',
            clip_on=True
        )

def _style_axes(ax, label_fontsize):
    ax.xaxis.grid(True, linestyle=':', linewidth=0.7, color='#bbb', zorder=0, alpha=0.5)
    ax.yaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#bbb')
    ax.tick_params(axis='y', pad=1.0)
    plt.yticks(fontsize=label_fontsize, color='#222')

def _add_prob_annotation(ax, label_names, probs, fontsize, pred_label, inverse=False):
    prob_lines = [f"{n}: {p:.2f}" for n, p in zip(label_names, probs)]
    prob_str = "\n".join(prob_lines)
    pred_name = label_names[pred_label].lower()
    if (pred_name == 'positive') != inverse:
        xy, ha = (0.98, 0.99), 'right'
    else:
        xy, ha = (0.05, 0.95), 'left'
    ax.annotate(
        f"Probabilities:\n{prob_str}",
        xy=xy, xycoords='axes fraction',
        fontsize=fontsize, color='#222', ha=ha, va='top',
        fontweight='medium',
        bbox=dict(boxstyle='round,pad=0.28', fc='#f8f8f8', ec='#888', lw=1.2, alpha=0.93)
    )

def _finalize_plot(fig, ax, margin=0.125):
    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    ax.set_xlim(xlim[0] - margin * x_range, xlim[1] + margin * x_range)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

def _add_input_text(fig, text, label_fontsize, bottom=0.01):
    max_chars = 180
    display_text = text if len(text) <= max_chars else text[:max_chars] + '...'
    fig.text(0.01, bottom, f"{display_text}", fontsize=label_fontsize, color='#333', ha='left', va='bottom', wrap=True, bbox=dict(facecolor='white', edgecolor='none', pad=2.5))

def explain_and_plot_transformer(model, tokenizer, text, label_names, save_path):
    if LimeTextExplainer is None:
        raise ImportError("LIME is not installed. Please install lime to use visualization.")
    from matplotlib import cm
    label_fontsize = 15
    header_fontsize = label_fontsize + 2
    pos_color = cm.Greens(0.6)
    neg_color = cm.Reds(0.5)
    explainer = LimeTextExplainer(class_names=label_names)
    def predict_proba(texts):
        import torch
        model.eval()
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            outputs = model(encodings['input_ids'], encodings['attention_mask'])
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    exp = explainer.explain_instance(text, predict_proba, num_features=20, labels=list(range(len(label_names))))
    pred_probs = predict_proba([text])[0]
    pred_label = int(pred_probs.argmax())
    pred_prob = float(pred_probs[pred_label])
    word_scores = dict(exp.as_list(label=pred_label))
    sorted_items = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    words = [w for w, _ in sorted_items]
    importances = [word_scores[w] for w in words]
    colors = _get_bar_colors(importances, pos_color, neg_color)
    fig, ax = plt.subplots(figsize=(6, min(1.1 + 0.36*len(words), 11)), facecolor='white')
    bar_height = 0.38
    bars = ax.barh(range(len(words)), importances, color=colors, edgecolor='none', zorder=3, height=bar_height)
    ax.axvline(0, color='#e0e0e0', linewidth=0.7, linestyle='-', zorder=2)
    ax.set_yticks(range(len(words)))
    # Fix yticklabels alignment and centering (match News-Sentiment style)
    ax.set_yticklabels(words, fontsize=label_fontsize, fontweight='medium', color='#222')
    ax.set_xlabel('LIME Score', fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel('Token', fontsize=label_fontsize, labelpad=10)
    ax.set_title(f"LIME Transformer Explanation for {label_names[pred_label]} Prediction (p={pred_prob:.2f})", fontsize=header_fontsize, fontweight='bold', pad=10, color='#222')
    _draw_bar_labels(ax, bars, label_fontsize)
    _style_axes(ax, label_fontsize)
    _finalize_plot(fig, ax)
    _add_prob_annotation(ax, label_names, pred_probs, label_fontsize, pred_label)
    fig.subplots_adjust(bottom=0.22, top=0.86, left=0.22, right=0.96)
    # _add_input_text(fig, text, label_fontsize, bottom=0.01)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return save_path

def explain_and_plot_lstm(model, vocab_builder, text, label_names, save_path, pred_probs=None, pred_label=None):
    from matplotlib import cm
    label_fontsize = 15
    header_fontsize = label_fontsize + 2
    pos_color = cm.Greens(0.7)
    neg_color = cm.Reds(0.7)
    neu_color = cm.Blues(0.6)
    import torch
    import numpy as np
    model.eval()
    # --- Tokenization: use same as prediction ---
    tokens = vocab_builder.preprocess([text])[0]
    input_ids = vocab_builder.encode([tokens], max_len=vocab_builder.max_len)
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    # --- Forward pass for prediction (no grad) ---
    with torch.no_grad():
        logits = model(input_tensor)["logits"]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label_pred = int(np.argmax(probs))
        pred_prob_pred = float(probs[pred_label_pred])
    # Use provided pred_probs/pred_label if available (from main prediction)
    if pred_probs is not None and pred_label is not None:
        probs = np.array(pred_probs)
        pred_label_pred = pred_label
        pred_prob_pred = float(probs[pred_label_pred])
    # --- Attribution: input x gradient ---
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    embeddings = model.embedding(input_tensor)
    embeddings.retain_grad()
    embeddings.requires_grad_()  # Only embeddings need gradients, not input_tensor
    lstm_out, _ = model.lstm(embeddings)
    attention_weights = torch.softmax(model.attention(lstm_out), dim=1)
    attended = torch.sum(attention_weights * lstm_out, dim=1)
    x = model.dropout(attended)
    x = torch.relu(model.fc1(x))
    x = model.dropout(x)
    logits = model.fc2(x)
    score = logits[0, pred_label_pred]
    model.zero_grad()
    score.backward()
    grad_tensor = embeddings.grad.squeeze()
    emb_tensor = embeddings.detach().squeeze()
    if grad_tensor.ndim == 2 and emb_tensor.ndim == 2:
        contribs = (emb_tensor * grad_tensor).sum(dim=1).cpu().numpy()
    elif grad_tensor.ndim == 1 and emb_tensor.ndim == 1:
        contribs = (emb_tensor * grad_tensor).cpu().numpy()
    else:
        contribs = grad_tensor.detach().cpu().numpy()
    real_token_count = len(tokens)
    contribs = contribs[:real_token_count]
    # --- Robust normalization ---
    # Compute baseline logit for pred_label_pred
    pad_token = getattr(vocab_builder, 'pad_token', None)
    token2idx = getattr(vocab_builder, 'word2idx', None)
    if pad_token and token2idx and pad_token in token2idx:
        pad_token_id = token2idx[pad_token]
    elif token2idx and '<pad>' in token2idx:
        pad_token_id = token2idx['<pad>']
    elif token2idx and len(token2idx) > 0:
        pad_token_id = list(token2idx.values())[0]
    else:
        pad_token_id = 0
    baseline_ids = [pad_token_id] * vocab_builder.max_len
    baseline_tensor = torch.tensor([baseline_ids], dtype=torch.long)
    with torch.no_grad():
        baseline_emb = model.embedding(baseline_tensor)
        baseline_lstm_out, _ = model.lstm(baseline_emb)
        baseline_attention_weights = torch.softmax(model.attention(baseline_lstm_out), dim=1)
        baseline_attended = torch.sum(baseline_attention_weights * baseline_lstm_out, dim=1)
        baseline_x = model.dropout(baseline_attended)
        baseline_x = torch.relu(model.fc1(baseline_x))
        baseline_x = model.dropout(baseline_x)
        baseline_logits = model.fc2(baseline_x)
        baseline_logit = baseline_logits[0, pred_label_pred].item()
    logit_diff = score.item() - baseline_logit
    contrib_sum = contribs.sum() if np.abs(contribs.sum()) > 1e-6 else 1e-6
    contribs = contribs * (logit_diff / contrib_sum)
    # --- Plot: tokens in original order, but reversed for y-axis (top = first token) ---
    colors = _get_bar_colors(contribs, pos_color, neg_color)
    if label_names[pred_label_pred] == 'neutral':
        colors = [neu_color if c > 0 else neg_color if c < 0 else '#cccccc' for c in contribs]
    tokens_rev = tokens[::-1]
    contribs_rev = contribs[::-1]
    colors_rev = colors[::-1]
    fig, ax = plt.subplots(figsize=(max(6, min(1.1 + 0.44*len(tokens), 13)), 1.1 + 0.44*len(tokens)), facecolor='white')
    bar_height = 0.44
    print(f"Tokens: {tokens}")
    print(f"Contributions: {contribs}")
    bars = ax.barh(range(len(tokens_rev)), contribs_rev, color=colors_rev, edgecolor='none', zorder=3, height=bar_height)
    ax.axvline(0, color='#e0e0e0', linewidth=0.7, linestyle='-', zorder=2)
    ax.set_xlabel("Input × Gradient (Token Attribution)", fontsize=label_fontsize, labelpad=10)
    ax.set_title(f"LSTM Token Importances\nPrediction: {label_names[pred_label_pred]} (p={pred_prob_pred:.2f})", fontsize=header_fontsize, fontweight='bold', pad=10, color='#222')
    ax.set_yticks(range(len(tokens_rev)))
    ax.set_yticklabels(tokens_rev, fontsize=label_fontsize, fontweight='medium', color='#222')
    _draw_bar_labels(ax, bars, label_fontsize)
    _style_axes(ax, label_fontsize)
    _finalize_plot(fig, ax)
    _add_prob_annotation(ax, label_names, probs, label_fontsize, pred_label_pred)
    fig.subplots_adjust(bottom=0.22, top=0.86, left=0.22, right=0.96)
    # _add_input_text(fig, text, label_fontsize, bottom=0.01)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return save_path
