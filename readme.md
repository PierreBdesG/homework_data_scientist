# Homework - Data Scientist

## Description

Ce repo permet de générer des images à partir de lignes de commande en utilisant des modèles compatible avec `Diffusers`. Il offre une interface simple et flexible pour personnaliser la génération d'images en fonction de divers paramètres. Utilisez la commande Python fournie pour générer des images avec vos prompts.

## Exemple d'utilisation

Vous pouvez générer une image en exécutant la commande suivante, le resultat sera stocker dans un dossier `output_images` :

```bash
python comfy2py.py --prompt="a beautiful forest in autumn" --neg-prompt="no snow, no fog"
```

## Arguments

`--prompt`
- **Type** : Union[List[str], str]
- **Description** : Le ou les prompts décrivant ce que vous souhaitez voir généré. 

` --neg-prompt`
- **Type** : Optional[Union[List[str], str]]
- **Description** : Le ou les prompts négatifs, qui permettent de spécifier des éléments à éviter dans l'image générée. 

` --steps`
- **Type** : Optional[int]
- **Description** : Le nombre de step. Plus ce nombre est élevé, plus le processus de génération est long et détaillé. La valeur par défaut est 50.

` --model-name`
- **Type** : Optional[str]
- **Description** : Le nom du modèle de diffusion à utiliser. La valeur par défaut est "runwayml/stable-diffusion-v1-5", mais vous pouvez utiliser un autre modèle compatible avec la class `DiffusionPipeline`.

` --height`
- **Type** : Optional[int]
- **Description** : La hauteur de l'image générée en pixels. La valeur par défaut est 512 (doit etre compatible avec le modèle).

` --width`
- **Type** : Optional[int]
- **Description** : La largeur de l'image générée en pixels. La valeur par défaut est 512 (doit etre compatible avec le modèle).

` --seed`
- **Type** : int
- **Description** : La valeur de la seed utilisée pour la génération de l'image. Si aucune seed n'est spécifiée, une seed aléatoire sera générée à chaque exécution.

` --cfg`
- **Type** : Optional[float]
- **Description** : Classifier-Free Guidance permet de contrôler la fidélité du modèle par rapport aux prompts. La valeur par défaut est 8.

` --output-dir`
- **Type** : Optional[str]
- **Description** : Le répertoire de sortie où l'image générée sera enregistrée. Par défaut, l'image est enregistrée dans `output_images`

` --num-images-per-prompt`
- **Type** : Optional[int]
- **Description** : Le nombre d'images à générer pour chaque prompt. La valeur par défaut est 1.

` --random-seed-after-every-gen`
- **Type** : Optional[bool]
- **Description** : Si activé, une nouvelle graine aléatoire sera générée après chaque image. `True` par default

` --sampler-name`
- **Type** : Optional[str]
- **Description** : Le nom de l'échantillonneur à utiliser pour la génération. `euler` par default.

` --scheduler`
- **Type** : Optional[str]
- **Description** : Le type de planificateur à utiliser. `normal` par default.

` --device`
- **Type** : Optional[str]
- **Description** : Le périphérique à utiliser pour la génération (ex: "cuda" pour GPU). Si laissé vide, le périphérique par défaut sera utilisé.






