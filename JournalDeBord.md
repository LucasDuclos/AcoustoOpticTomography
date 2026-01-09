# Journal de bord thèse

## 15 Octobre 2025

Je suis allé à Langevin réalisé des manips avec François sur leur montage photoréfractif.
J'ai ramené un bloc de PVA relativement épais (penser à demander à François les dimensions exactes). Nous avons insérer dedans une gaine de cable électrique. Sur l'échographe on remarque quelle ne perturbe pas trop la propagation des US.

**Remarque:**
La sonde SL10-2 de l'aixplorer est abimée au centre on voit une ombre qui part du milieu de la sonde.

Comme il me l'avait indiqué, l' amplitude des signaux Acousto-optique en onde plane est nettement supérieure à celle des ondes structurées.
Grosse différence entre le montage à Langevin et celui d'Orsay, François utilise une fribre optique avec un zoom à l'extrémité qui lui permet de régler la taille du faisceau à l'entré de l'objet et d'illuminé de manière homogène le fantôme.

<img width="2082" height="580" alt="image" src="https://github.com/user-attachments/assets/36fbb4fb-4693-4f62-a1bd-231587e94aba" />

Figure 1: Reconstruction expérimentale d'un fantôme de PVA avec inclusion. À gauche: OF, au milieu: OP, à droite OS. 
---
De mon point de vu, la gaine en caoutchouc est trop grosse, mal positionné vis à vis du laser. 


----
## 17 Octobre 2025

#### Analyse des signaux AO acquis le 15 octobre 2025

Après analyse des signaux AO acquis lors de la manip à Langevin, 

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/46c83b69-0412-4ec4-9211-b8f3a50c957d" />

Figure 2: Signaux acousto-optique issus de trois acquisitions différentes.
---

On se rend compte que les mêmes emissions acoustiques, onde plane 0° et onde structurée (tous les piezos allumés donc revient à une onde plane) à 0°, ne donne pas les mêmes amplitudes et niveau de bruit.

Avec les paramètres suivant:

- Onde Plane --> Fréquence sonde : 3 MHz, Nombre Hémicycle : 4, Tension : 50 V, Nombre de répétitions : 500 --> ffffffffffffffffffffffffffffffffffffffffffffffff_000
- Onde structurée --> Fréquence sonde : 3 MHz, Nombre Hémicycle : 4, Tension : 50 V, Nombre de répétitions : 500 --> ffffffffffffffffffffffffffffffffffffffffffffffff_000
- Onde Focus --> Fréquence sonde : 3 MHz, Nombre Hémicycle : 4, Tension : 50 V, Nombre de répétitions : 500 


J'obtiens ces résultats: 

- OS: Signal 20, moyenne = 23.08 mV
- OF: Signal 80, moyenne = 6.92 mV
- OP: Signal 20, moyenne = 31.15 mV
- Pourcentage de différence : 34.99 % entre le signal AO en OP et en OS
  
Avec :

- Écart-type OS (signal 20) : 0.0195 mV
- Écart-type OP (signal 20) : 0.0266 mV
- Différence de bruit : 36.25 % entre le signal AO en OP et en OS
---
### Reconstructions à partir des données expérimentales.

Je rencontre beaucoup de soucis avec les algos implémentés par Kaiyuan. Les hyper paramètres sont très sensibles et le faible nombre d'itérations avant divergence complique leur réglage.
Commençons par le code le plus simple, le MLEM, il nécessite pas de réglage d'hyperparamètres.

Le MLEM codé par kaiyuan présente des problème de stabilité numérique.

Avec ondes planes, les paramètres des émissions sont détaillés ci-dessous (angles et structurations)

<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/8095e9e5-4274-430f-b6fe-991fea16d280" />

Les résultats sont les suivants

<img width="500" height="900" alt="image" src="https://github.com/user-attachments/assets/6ecc849d-cb37-4984-9bbe-aeac68075684" />

Figure 3: Reconstruction MLEM avec Kwave et aot-biomaps, pour différentes itérations.
---

On voit rapidement sur les reconstructions, plus on augmente le nombre d'itération et plus l'algo est instable. Claude dit que c'est sans doute à cause d'un seuil au dénominateur trop faible. 
J'ai prévu de comparer les résultats depuis CASTor

---
## 20 Octobre 2025

À faire:
- afficher les résultats avec CASToR et kwave.
- Mettre à jour dans la librairie le chargement des reconstructions.
- Mettre à jour dans la librairie le reconstruction CASToR.

---
J'ai pas mal galérer sur le code du chargement des signaux AO au niveau de la librairie. Maintenant les reconstructions kwave et field2 marchent sur CASToR et AOT_biomaps.

J'ai modifié le code de Kaiyuan, il n'avait pas mis comme dans CASToR un seuil pour le dénominateur (il avait rajouté un torch.tiny au valeur au numérateur pour éviter la division par 0). Même avec ça, on visualisait des aberations dans la reconstruction. J'ai donc établi un seuil de 1e-2 au dénominateur pour éliminer complétement les artefacts lié à la reconstruction.

<img width="1014" height="570" alt="image" src="https://github.com/user-attachments/assets/bee191b3-1c5c-4a46-8b46-3fcb81a5ab2b" />


En ondes planes 

avec field2:
<img width="1445" height="1136" alt="image" src="https://github.com/user-attachments/assets/c5bdd04c-fdbd-405b-84ff-07ce05fb5876" />

![recon_field2](https://github.com/user-attachments/assets/a05ecf46-74c5-4a68-99b5-90e167e2d071)


avec kwave:
<img width="1445" height="1136" alt="image" src="https://github.com/user-attachments/assets/3508d349-2014-458a-8e36-e3a09a4da28c" />

![recon_kwave](https://github.com/user-attachments/assets/2ef47a12-41fb-4a99-b8ac-aacccb1bd12b)

CASToR ressort les mêmes résultats.

Kwave reconstruit mieux:

<img width="1574" height="574" alt="image" src="https://github.com/user-attachments/assets/c240261f-0fc4-4bfb-ad4b-33d6f82a7057" />

<img width="423" height="455" alt="image" src="https://github.com/user-attachments/assets/2d5da029-0157-4472-95c7-e8d6264e3e3b" />

---
## 21 Octobre 2025

Comme expliqué dans la figure 2 du 17 octobre, les signaux acousto-optiques mesurés en onde structurées ne sont pas convaincants. J'ai prévu de refaire des manips cette après midi à Orsay. 

Manip:
J'ai placé la fibre optique en sortie du beamsplitter. J'ai placé une lentille de 50 mm de focal entre ma fibre et le cube pour minimiser le diamètre du faisceau à l'entrée de la fibre. J'ai au mieux après alignement 11.8% d'éfficacité. On s'en contentera.

## 02 Décembre 2025

François est venu au laboratoire le mois dernier pour bien paramétrer le montage optique. Normalement tout fonctionne maintenant.
J'ai fait 2 acquisitons.
Ci dessous, les reconstructions focalisées des fantôme de PVA (fin : 1 cm et épais : 3.5cm)
<img width="1014" height="416" alt="image" src="https://github.com/user-attachments/assets/d16810a7-ef9b-45d1-b47d-e833936c9f06" />

Pour les reconstructions tomographique j'utilise une implémentation GPU cuda perso pour sparse les matrices systèmes afin de les stocker en mémoire VRAM du GPU. J'ai implémenté un algo GPU Chambolle pock LS-TV-Tikhonov pour reconstruire \lambda avec les matrices sparses. Finalement je suis resté sur uniquement LS-Tikhonov, la TV ne marche pas bien et Tikhonov suffit pour lisser correctement.

<img width="511" height="371" alt="image" src="https://github.com/user-attachments/assets/1011ebf4-b4ad-412f-b96a-7ca332717dc5" />

Voici la reconstruction du fantôme épais avec CP LS-TV. L'image est bien reconstruite mais on voit un artefact (en forme de V) qui induit une accumulation de signal en dehors de la tache. Si je mets à saturation on visualise: 

<img width="626" height="473" alt="image" src="https://github.com/user-attachments/assets/98f0adde-2a25-4c3b-9017-ec131a8da7d1" />

Les deux artefacts sont bien présents. J'ai essayé avec un algo MLEM classique : le bruit est tres présent mais on visualise aussi cet artefact:

<img width="552" height="417" alt="image" src="https://github.com/user-attachments/assets/e928a0ff-d44b-46fa-9c7d-7f9e1bb57b1e" />

Sur des reconstructions simulées avec beaucoup de bruit on n'a pas ces artefacts, ni en MLEM ni en CP. Voici en CP ci dessous:

<img width="858" height="349" alt="image" src="https://github.com/user-attachments/assets/37faf603-a580-45b7-a747-73749a3e9660" />

C'est la même image à gauche et à droite sauf qu'à droite je sature l'échelle de couleur et on voit bien qu'il n'y a pas cette structure en V.
La structure en V n'est pas induite par l'algo en présence de bruit.
À mon avis, elle provient:
  - Soit d'un mauvais alignement entre les données mesurées et la matrice système (retards...) et la je sais pas trop comment vérifier
  - Soit les paramètres acoustiques de la simulation K-wave ne sont pas réaliste
  - Soit la simulation K-wave en elle même n'est pas réaliste.
Je vais comparer avec field2 pour essayer d'y voir plus clair.

## 09 Janvier 2025

<img width="984" height="985" alt="image" src="https://github.com/user-attachments/assets/8b3cae6a-e494-4998-a3e5-e894b0e8da78" />




