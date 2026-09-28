# Actual PydanticAI comparisons

Ordinary generation and marked client output from the same local Qwen/MLX setup. All outputs retained. These examples are integration evidence, not general quality or detector acceptance.

## english

Prompt: Explain in one sentence why a seed needs water.

Ordinary:

```text
A seed needs water to activate the metabolic processes that allow it to germinate and grow into a plant.
```

Marked (PydanticAI output):

```text
A seed needs water to activate the metabolic processes that initiate germination and allow the embryo to grow into a plant.
```

## french

Prompt: En français, écris une phrase demandant à Maya de garder la sauvegarde jusqu’à la vérification de la restauration.

Ordinary:

```text
Demandez à Maya de garder la sauvegarde jusqu’à la vérification de la restauration.
```

Marked (PydanticAI output):

```text
Pourrais-tu demander à Maya de garder la sauvegarde jusqu’à la vérification de la restauration ?
```

## spanish

Prompt: En español, escribe una frase pidiendo a Maya que espere mi aprobación antes de publicar el documento.

Ordinary:

```text
Por favor, espera mi aprobación antes de publicar el documento, Maya.
```

Marked (PydanticAI output):

```text
Por favor, espera mi aprobación antes de publicar el documento, Maya.
```

## structured

Prompt: Return JSON using only these facts: Maya reviews the draft by Tuesday 09:30. Vincent approves publication. Publication cannot proceed without approval. Use fields reviewer, deadline, approver, publish_without_approval. Preserve the names and exact deadline.

Ordinary:

```text
{
  "reviewer": "Maya",
  "deadline": "Tuesday 09:30",
  "approver": "Vincent",
  "publish_without_approval": false
}
```

Marked (PydanticAI output):

```text
{
  "reviewer": "Maya",
  "deadline": "Tuesday 09:30",
  "approver": "Vincent",
  "publish_without_approval": false
}
```
