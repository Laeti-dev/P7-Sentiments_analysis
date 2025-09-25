Projet Air Paradis : Anticiper les "Bad Buzz" grâce à l'IA

  Par : Laetitia Ikusaza, Ingénieure IA, Marketing Intelligence Consulting (MIC)

  Bonjour à tous,

  Dans deux semaines, nous avons un rendez-vous crucial avec Mme Aline, la directrice marketing de
  notre nouveau client, la compagnie aérienne "Air Paradis". Comme vous le savez, leur réputation
  sur les réseaux sociaux est un enjeu majeur. Notre mission ? Créer un produit d'Intelligence
  Artificielle capable de détecter et d'anticiper les vagues de commentaires négatifs, ou "bad
  buzz".

  Cet article vise à vous présenter, de manière simple, notre démarche pour construire cette
  solution, en mettant l'accent sur notre approche MLOps qui garantit la qualité et la fiabilité de
  notre travail.

  Étape 1 : Comprendre le terrain de jeu - L'analyse des données

  Tout projet IA commence par les données. Pour apprendre à une machine à reconnaître un sentiment
  (positif ou négatif), il faut d'abord lui en montrer des milliers d'exemples.

  Nous avons utilisé un jeu de données public contenant 1,6 million de tweets, chacun étant déjà
  étiqueté comme "positif" ou "négatif".

  Notre première mission, a été d'explorer ces données pour :
   1. Vérifier leur qualité : Les données étaient propres, sans valeurs manquantes, mais contenaient quelques doublons que nous avons nettoyés.
   2. Comprendre la répartition : Le jeu de données était parfaitement équilibré, avec autant de tweets positifs que négatifs. C'est une base idéale pour entraîner un modèle juste.
   3. Identifier des tendances : Nous avons analysé les mots les plus fréquents dans chaque catégorie.Par exemple, les tweets négatifs contenaient souvent des mots comme "sad", "miss", "can't",tandis que les tweets positifs utilisaient "love", "good", "thanks".

  Cette exploration nous a donné une compréhension fine du langage utilisé sur les réseaux sociaux.


  Visuel des mots les plus fréquents : à gauche les négatifs, à droite les positifs.

  Étape 2 : Apprendre à la machine à lire - La préparation des textes

  Un ordinateur ne comprend pas les mots, il ne comprend que les chiffres. L'étape suivante a donc consisté à "traduire" les tweets en un format numérique.

  Ce processus de nettoyage, appelé preprocessing, inclut :
   - Le nettoyage : Suppression des liens web, des mentions (@utilisateur) et autres "bruits".
   - La normalisation : Mise en minuscule, correction de l'argot ("u" -> "you") et des fautes de frappe.
   - La simplification (Lemmatisation) : Ramener les mots à leur racine (par exemple, "running" et "ran" deviennent "run").

  Une fois les textes nettoyés, nous les avons transformés en vecteurs numériques grâce à deux
  techniques :
   - Bag-of-Words (BoW) : Une méthode simple qui compte la fréquence de chaque mot dans un tweet.
   - TF-IDF : Une version plus intelligente qui donne plus de poids aux mots importants et rares.

  Étape 3 : Nos premiers modèles - L'approche classique

  Avec des données numériques propres, nous avons entraîné nos premiers modèles de classification.
  Nous avons commencé avec des algorithmes de Machine Learning "classiques" comme la Régression
  Logistique et le Naive Bayes.

  Pour chaque expérience, nous avons mesuré plusieurs indicateurs de performance :
   - L'Accuracy (Précision globale) : Le pourcentage de prédictions correctes.
   - Le Rappel (Recall) : La capacité du modèle à retrouver tous les tweets négatifs. C'est un
     indicateur crucial pour ne manquer aucun début de "bad buzz".
   - Le F1-Score : Un score qui équilibre la précision et le rappel.

  > Notre démarche MLOps : Chaque test, chaque modèle et chaque résultat ont été rigoureusement enregistrés grâce à MLflow. C'est notre "carnet de bord" intelligent. Il nous permet de comparer objectivement les performances, de garantir que nos résultats sont reproductibles et de choisir le meilleur modèle en toute confiance.

  Les premiers résultats étaient prometteurs, avec une précision d'environ 77%.

  Étape 4 : Vers l'excellence - Les réseaux de neurones

  Pour aller plus loin, nous nous sommes tournés vers les réseaux de neurones, des modèles plus complexes inspirés du cerveau humain.

  La grande innovation ici est l'utilisation des Word Embeddings (comme Word2Vec ou GloVe). Au lieu de simplement compter les mots, cette technique capture leur sens et leur contexte. Les mots ayant des significations similaires sont représentés par des vecteurs numériquement proches. C'est ce qui permet au modèle de comprendre des nuances comme la différence entre "je ne suis pas content"
  et "je suis très content".

  Les résultats ont été à la hauteur de nos attentes. Les modèles neuronaux, plus fins dans leur analyse, ont montré des performances supérieures, notamment en termes de rappel, ce qui est essentiel pour la mission d'Air Paradis.

  Conclusion et prochaines étapes

  En partant d'un océan de tweets bruts, nous avons construit et testé une série de modèles de plus en plus sophistiqués. Notre approche MLOps, avec le suivi systématique via MLflow, nous assure une traçabilité et une qualité irréprochables.

  Nous disposons aujourd'hui d'un modèle performant, prêt à être déployé. Lors de notre rendez-vous avec Mme Aline, nous serons en mesure de lui présenter une solution robuste, éprouvée et prête à être intégrée dans ses outils de veille pour protéger l'image d'Air Paradis.

  Ce projet est un excellent exemple de la manière dont MIC allie expertise en IA et rigueur
  méthodologique pour créer de la valeur tangible pour nos clients.
