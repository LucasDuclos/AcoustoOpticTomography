# Journal de bord thèse

### 15 octobre 2025

Je suis allé à Langevin réalisé des manips avec François sur leur montage photoréfractif.
J'ai ramené un bloc de PVA relativement épais (penser à demander à François les dimensions exactes). Nous avons insérer dedans une gaine de cable électrique. Sur l'échographe on remarque quelle ne perturbe pas trop la propagation des US.

**Remarque:**
La sonde SL10-2 de l'aixplorer est abimé au centre on voit une ombre qui part du milieu de la sonde.

Comme il me l'avait indiqué, l' amplitude des signaux Acousto-optique en onde plane est nettement supérieure à celle des ondes structurées.
Grosse différence entre le montage à Langevin et celui d'Orsay, François utilise une fribre optique avec un zoom à l'extrémité qui lui permet de régler la taille du faisceau à l'entré de l'objet et d'illuminé de manière homogène le fantôme.

<img width="2082" height="580" alt="image" src="https://github.com/user-attachments/assets/36fbb4fb-4693-4f62-a1bd-231587e94aba" />

De mon point de vu, la gaine en caoutchouc est trop grosse, mal positionné vis à vis du laser. 

----
----

### 17 Octobre 2025

Après analyse des signaux AO acquis lors de la manip à Langevin, 

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/46c83b69-0412-4ec4-9211-b8f3a50c957d" />

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

Le MLEM codé par kaiyuan présente des problème de stabilité numérique.

<img width="500" height="900" alt="image" src="https://github.com/user-attachments/assets/6ecc849d-cb37-4984-9bbe-aeac68075684" />
