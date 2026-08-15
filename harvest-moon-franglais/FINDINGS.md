# Patch franglais pour Harvest Moon: Friends of Mineral Town (GBA)

Étude de faisabilité — mesures réalisées sur la ROM, pas des estimations.

Objectif : un patch qui démarre en français et introduit progressivement
l'anglais, pour apprendre la langue en jouant.

## Cible

| | |
|---|---|
| Jeu | Harvest Moon: Friends of Mineral Town (USA), GBA, 2003 |
| SHA1 | `a2fc3574f0a65a4fcf7682fb274b9d7eebdef963` |
| Taille | 8 Mio |
| Espace libre | à partir de `0x0875C244` (~655 Kio), extension possible (cible émulateur) |

FoMT plutôt que *More* FoMT parce que c'est la seule des deux à être décompilée.
Le corpus de texte restera portable vers MFoMT plus tard : `mary` nomme les
callables FoMT avec les noms MFoMT précisément pour faciliter la comparaison.

## Outillage amont (tout fonctionne, vérifié)

- [StanHash/fomt](https://github.com/StanHash/fomt) — décompilation, rebuild à
  l'octet près depuis `baserom.gba` (`make compare`).
- [StanHash/mary](https://github.com/StanHash/mary) — compilateur/décompilateur
  de scripts d'événements. Build OK (`cargo build --release`), décompilation et
  recompilation testées.
- [StanHash/FOMT-DOC](https://github.com/StanHash/FOMT-DOC) — documentation des
  internals (boîtes de texte, layout ROM, compression).

## Volume réel du texte

Extraction complète : **1328 scripts, 8368 chaînes, 0 échec**.

| Bloc | Scripts | Chaînes | Mots |
|---|---:|---:|---:|
| Jeu (dialogues, événements, festivals) | 1150 | 7372 | **70 869** |
| Bibliothèque + émissions de télé | 17 | 996 | 85 193 |
| **Total** | **1167** | **8368** | **156 062** |

**Le résultat structurant de l'étude : 17 scripts contiennent 62 % du texte du
jeu**, et ce sont les livres de la bibliothèque et les programmes télé
(`Life on the Farm BEGINNER/ADVANCED`, `My Dear Princess`, `Fairy and Me`,
`Star Lily, Bandit Girl`, `MECHABOT GENESIS`, `Card Collector Chisato`…).

Conséquences :

1. Le texte **nécessaire pour jouer** ne fait que ~71 000 mots, pas 156 000.
   Le devis de traduction est divisé par deux.
2. La bibliothèque est un **lecteur gradué déjà découpé**, opt-in, sans
   pression, et déjà partitionné en scripts séparés. C'est le véhicule
   pédagogique idéal : le joueur qui veut plus d'anglais y va de lui-même.
   Le jeu propose même déjà des livres « BEGINNER » et « ADVANCED ».

Déduplication : 7004 chaînes uniques sur 8368 (83 %), soit 1364 occurrences
de travail économisées (`Yes` ×33, `No` ×33, `It's locked...` ×25…).

## Contraintes techniques mesurées

### Largeur d'affichage : 28 caractères, en dur

Mesuré sur les 8368 chaînes : 7563 lignes font exactement 28 caractères, et la
distribution s'effondre au-delà (36 lignes à 29, 18 à 30, 3 à 31, 1 à 32 — et
ces cas sont probablement un artefact de notre estimation à 8 caractères pour
les noms substitués). **28 est le budget par ligne, 3 lignes par boîte.**

C'est la contrainte dominante du projet : le français est ~20 % plus long que
l'anglais. Elle joue toutefois *en faveur* du concept — chaque mot anglais
réintroduit rend de la place.

### Longueur des chaînes : non contrainte

Chaque script est un conteneur RIFF avec ses propres chunks `CODE` / `STR ` /
`JUMP` : **un pool de chaînes privé par script**, indexé par offsets
(cf. `AScriptEngine::GetString`, `src/script_engine.cc:605`). Comme `mary`
recompile le script entier, il régénère le pool. Aucun recalcul de pointeurs,
aucune limite de longueur héritée de l'original. Seuls les 28 caractères par
ligne comptent.

### Codes de contrôle

Le jeu suit l'ASCII (`FOMT-DOC/TextBoxes.txt`), confirmé sur le corpus :

| Code | Rôle | Occurrences |
|---|---|---:|
| `\x05` | attendre la touche A | 9 197 |
| `\x0A` | line feed | 38 056 |
| `\x0C` | vider la boîte | 1 430 |
| `\x0D` | retour chariot | 38 056 |

Substitutions `\xFF` + sélecteur : 13 sélecteurs distincts, dont `\xFF!` (nom du
joueur, 468 occurrences) et `\xFF%` (nom d'animal, 159). **À préserver
verbatim** — et attention à l'élision française (« de \xFF! » vs « d'\xFF! »),
qui n'a pas d'équivalent en anglais et demandera des tournures évitant le
problème.

### Police et accents — le seul point ouvert

91 des 95 caractères ASCII imprimables sont utilisés ; seuls `\`, `{`, `}`, `~`
sont libres. Insuffisant pour é è ê à â î ô û ù ç et leurs majuscules.

En revanche le corpus emploie **101 valeurs d'octets distinctes au-dessus de
`0x7F`** : la police contient largement plus que l'ASCII. Il reste à localiser
la table de glyphes de FoMT (`FOMT-DOC/ROMLayout.txt` donne `0x089AB014` pour
MFoMT) et à vérifier ce qui s'y trouve déjà. C'est le blocage historique qui a
tué la tentative de traduction FR de 2008 — la décompilation le rend soluble,
mais il doit être traité **en premier**.

## Ce qu'on sait dire, et quand

C'était la question ouverte. Réponse : le contenu est très largement
attribuable, par trois voies cumulables.

**1. Le locuteur est extractible mécaniquement.** Dans les scripts décompilés,
chaque `TalkMessage` est précédé d'un `Proc02D(n)` qui désigne l'entité qui
parle. Exemple réel (script 760, conversation Sasha ↔ Manna) :

```
    Proc02D(1)   TalkMessage(MESSAGE_0)   // "You and Jeff have been married..."
    Proc02D(18)  TalkMessage(MESSAGE_1)   // "Yes."
```

On peut donc produire un corpus **indexé par personnage** sans reverse
engineering supplémentaire.

**2. La table de scripts est structurée par personnage.** 20 % des scripts
citent nommément un membre du cast, et ces scripts se concentrent à **48 % en
moyenne dans leurs 3 principales plages d'IDs contiguës** (Ann : 915-924,
906-911 ; Popuri : 898-905, 889-894 ; Elli : 945-961…). La table n'est pas
ordonnée au hasard.

**3. Le contenu se date tout seul, partiellement.** Un test naïf par mots-clés
(saisons, festivals, mariage, mine, tutoriel) marque 26 % des scripts. Faible
seul, utile en complément.

**Ce qui reste inconnu** : les *conditions de déclenchement* des scripts ne sont
pas rétro-ingéniérées (StanHash : « It is not yet well known how we can
configure how scripts are triggered »). On sait donc **qui dit quoi**, pas
**à quelle date exacte ça se déclenche**. Pour un patch pédagogique qui n'a
besoin que de 3 à 5 paliers, ce n'est pas bloquant — et le croisement avec la
documentation communautaire (Fogu / Ushi no Tane, qui documente FoMT événement
par événement) comble le reste sans toucher à la ROM.

**Conséquence de conception** : puisque l'attribution par *personnage* est
fiable et la datation par *événement* ne l'est pas, la progression doit se faire
**personnage par personnage** plutôt que date par date. C'est aussi meilleur
narrativement — Kai, le saisonnier de passage, peut parler anglais dès le
départ.

## Architecture retenue

Point d'accroche unique, `src/script_engine.cc:605` :

```cpp
char const * AScriptEngine::GetString(u32 id) const
{
    if (id <= string_count)
        return string_pool + string_offset_table[id];
    return "Error";
}
```

**Tout le texte des 1328 scripts passe par cette fonction de 5 lignes, déjà
décompilée.** On y branche une table de variantes indexée par
`(script_id, string_id, niveau)` — `ScriptEngine::LoadById` est juste en
dessous, donc le script courant est connu. Bénéfices :

- une seule fonction modifiée ;
- **zéro modification des 1328 scripts**, donc `make compare` continue de
  valider le reste de la ROM à l'octet près ;
- les chaînes identiques entre niveaux ne sont stockées qu'une fois.

Les noms d'objets suivent un autre chemin — ils sont **déjà des fichiers texte
éditables** dans la décompilation (`data/item/{tool,food,product,article}.def`,
348 entrées, 694 chaînes) — et demandent juste un tableau indexé par niveau.

**Stockage du niveau** : la bibliothèque SRAM n'est pas décompilée
(`asm/code_lib_sram.s`, checksum en assembleur). Ne pas toucher au format de
sauvegarde : écrire le niveau dans une zone SRAM inutilisée avec son propre
checksum, ce qui garde la compatibilité des sauvegardes vanilla dans les deux
sens.

**Réglage in-game plutôt qu'au démarrage** : la doc de l'écran-titre est
embryonnaire et écrite pour MFoMT. Mettre le réglage dans la télé de la maison
ou le journal est moins cher techniquement et meilleur pédagogiquement — le
joueur ajuste au fil des saisons au lieu de s'engager à l'aveugle.

**Touche « je n'ai pas compris »** : `TextBoxInterpreter` possède déjà un champ
`+10 | pointer to another null-terminated string (takes priority over [+08])`.
Le moteur sait donc déjà substituer une chaîne à la volée : réafficher la ligne
courante en français intégral sur L ou R est quasi natif. C'est la
fonctionnalité qui supprime la peur de rater du contenu — celle qui fait
abandonner ce genre de patch.

## Découpage

| Phase | Contenu | Moteur ? |
|---|---|---|
| **P0** | Police + accents ; pipeline texte ; objets + printemps an 1, un niveau figé | non |
| **P1** | Hook `GetString` + niveau en SRAM + réglage in-game | oui, ciblé |
| **P2** | Touche « aide » (réaffichage en français) | oui, petit |
| **P3** | Bibliothèque en lecteur gradué | non |
| **P4** | Port MFoMT via `mary` | — |

P0 ne demande aucune modification du moteur et valide tout le reste.

## Outils de ce dépôt

| Script | Rôle |
|---|---|
| `tools/extract_strings.py` | extrait les 8368 chaînes de la ROM en TSV |
| `tools/analyze.py` | volume, largeur d'affichage, codes de contrôle, databilité |
| `tools/cluster.py` | regroupement par personnage, densité par tranche d'IDs |

```sh
python3 tools/extract_strings.py baserom.gba > strings_en.tsv
python3 tools/analyze.py strings_en.tsv
python3 tools/cluster.py strings_en.tsv
```

La ROM et les dumps de texte ne sont pas versionnés (cf. `.gitignore`) : ce sont
des données de jeu sous copyright. Seuls les outils et l'analyse le sont. La
distribution finale se fera sous forme de patch (BPS/IPS), jamais de ROM.
