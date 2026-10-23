# 04 — Small model for accurate tool use

**Build fourth.** Reuse project 2's sandbox tools and project 1's serving/measurement harness.

## Objective

Test whether a smaller specialized open-weight model can approach a larger general model's success on a narrow tool-use task while using fewer resources. Make the project an experimental study with executable scoring.

## Why this fits Amazon and Microsoft

[Amazon Nova Forge](https://aws.amazon.com/nova/forge/) supports model customization. [Microsoft's Phi reasoning work](https://www.microsoft.com/en-us/research/video/pushing-boundaries-of-complex-reasoning-in-small-language-models/) studies strong reasoning in smaller models. Your experiment uses a locally supported open-weight model; Nova Forge access is not necessary.

2028+ assessment: specialization, data quality, evaluation, and quality/resource tradeoffs are likely to stay useful. Matching these themes does not imply reproducing either company's research.

## Minimum viable scope

- One narrow task domain, such as sandbox service diagnostics.
- Approximately 5–10 tools with strict argument schemas and observable effects.
- One supported 3–8B base model and parameter-efficient supervised fine-tuning.
- A dataset builder, independent task executor, and held-out evaluation.
- Comparisons with the unchanged base model, prompting improvements, and a larger model if feasible.

## Proposed stack and architecture

PyTorch, Transformers, PEFT/LoRA, pinned datasets/configuration, and the local inference server. Check compatible packages and recipes using [NVIDIA's Spark fine-tuning guide](https://build.nvidia.com/spark/pytorch-fine-tune/overview).

```text
Task generator -> Split definitions -> Training examples -> Fine-tuning
                          |
                    Held-out tasks -> Model calls -> Sandbox executor -> Score
```

Start with single-tool tasks, then add multi-step tasks. The executor determines success from resulting state, not from the model's assertion.

## Implementation stages

1. **Task definition:** define tools, valid outputs, and independent success conditions. Exit when a hand-written policy can be evaluated.
2. **Leak-resistant splits:** separate task templates/entities/scenarios before generating examples. Deduplicate and audit overlaps.
3. **Baselines:** evaluate the base model and a stronger prompting strategy under fixed tool and retry budgets.
4. **Pilot fine-tune:** use a small subset to confirm training and model reload work on the GX10. Then expand the dataset.
5. **Full comparison:** evaluate a LoRA-trained model against baselines on untouched tasks.
6. **Ablations:** vary one factor at a time, such as data quantity, data quality, or adapter configuration. Repeat key experiments when compute permits.

## Data plan

Begin with hundreds of validated examples; scale to thousands only after the generator and scorer are trustworthy. Synthetic tasks should have varied wording, tool schemas, entities, and failure conditions. Use deterministic validators and manual spot checks.

If using model-generated examples or teacher outputs, verify licensing/terms, record provenance, and prevent the teacher from seeing held-out cases. A licensed deterministic task generator can avoid paid teacher dependencies entirely.

## Evaluation

Report task completion, tool-selection accuracy, argument validity, unnecessary/unauthorized calls, retries, latency, and memory. Include unfamiliar templates, unavailable tools, ambiguous requests, and no-action cases.

Use identical tools and evaluation budgets across comparisons. Report uncertainty or variation across repeated runs/seeds where feasible. Track whether specialization harms a separate small general-capability set. Make a larger-model comparison sequentially if memory is tight; use the same task definitions.

## Important tests

- Invalid tool arguments fail before execution.
- The scorer rejects plausible-looking calls that produce the wrong state.
- Training data cannot include evaluation answers or near-duplicate templates inadvertently.
- Adapter loading reproduces the evaluated checkpoint's results.
- No-action and missing-information tasks are scored explicitly.

## GX10 constraints

Start with LoRA and modest sequence lengths/batches. Measure peak memory with a pilot run before expanding. Quantized training is an optional experiment only if supported on your exact software/hardware combination. Avoid full large-model training from scratch.

## Definition of done

Publish a reproducible study with baselines, at least one controlled ablation, held-out results, and failure analysis. A negative result is useful if you explain it with sound measurements.

Deliver a dataset card, split definitions, training/evaluation scripts, configurations, adapter/model card, results, and experimental write-up. Résumé template: “Fine-tuned a [size] model for [task]; measured [actual change] versus [baseline] on [held-out split], including latency and error analysis.”
