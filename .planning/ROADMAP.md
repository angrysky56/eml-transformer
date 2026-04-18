# Roadmap: Layer 2

## Milestone: Learned Program Generator

### Phase 1: Data Generation & Tokenization
- [ ] **Phase 10.1: Data generator module**
  - Implement `src/eml_transformer/layer2/dataset.py`.
  - Enumerate RPN programs up to depth 2.
  - Verify with pair count and spot-checks.
- [ ] **Phase 10.2: Layer 2 Tokenizer**
  - Implement `src/eml_transformer/layer2/tokenizer.py`.
  - Vocabulary includes 27 catalog names as atoms.
  - Verify with round-trip tests.

### Phase 2: Training Infrastructure
- [ ] **Phase 10.3: PyTorch Dataset & Collation**
  - Implement `src/eml_transformer/layer2/torch_dataset.py`.
  - Signature encoding (12-dim real vector).
  - Verify batch shapes and collation.
- [ ] **Phase 10.4: Model Architecture**
  - Implement `src/eml_transformer/layer2/model.py`.
  - Cross-attending decoder (0.5M-1M params).
  - Verify forward pass and generation.

### Phase 3: Training & Generalization
- [ ] **Phase 10.5: Supervised Training Loop**
  - Implement `src/eml_transformer/layer2/train.py`.
  - Split by depth (Train depth 1, Val depth 2).
  - Verify signature match metric.
- [ ] **Phase 10.6: (Optional) RL Fine-tuning**
  - Implement `src/eml_transformer/layer2/rl.py`.
  - Use verifier as reward signal.

### Phase 4: Finalization
- [ ] **Phase 10.7: Evaluation & Documentation**
  - Final benchmarking on depth-2 generalization.
  - Update README with Layer 2 results.
