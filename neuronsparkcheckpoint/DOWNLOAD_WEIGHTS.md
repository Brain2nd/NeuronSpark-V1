# Downloading the model weights

The model weights (`model-00001-of-00002.safetensors` and
`model-00002-of-00002.safetensors`, ~2.2 GB total in bf16) are not bundled
inside this OpenReview zip.

Fetch them from the anonymous repository hosted via anonymous.4open.science:

```
https://anonymous.4open.science/r/NeuronSpark
```

Direct file URLs (resolve to the underlying Git LFS objects):

```
https://anonymous.4open.science/api/repo/NeuronSpark/file/neuronsparkcheckpoint/model-00001-of-00002.safetensors
https://anonymous.4open.science/api/repo/NeuronSpark/file/neuronsparkcheckpoint/model-00002-of-00002.safetensors
```

Quick command:

```bash
cd neuronsparkcheckpoint
curl -L -o model-00001-of-00002.safetensors \
  "https://anonymous.4open.science/api/repo/NeuronSpark/file/neuronsparkcheckpoint/model-00001-of-00002.safetensors"
curl -L -o model-00002-of-00002.safetensors \
  "https://anonymous.4open.science/api/repo/NeuronSpark/file/neuronsparkcheckpoint/model-00002-of-00002.safetensors"
```

Once both shards are in `neuronsparkcheckpoint/`, the directory becomes a
self-contained HuggingFace artifact and can be loaded with
`AutoModelForCausalLM.from_pretrained("./neuronsparkcheckpoint", trust_remote_code=True)`
as described in the top-level `README.md`.
