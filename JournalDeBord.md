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


