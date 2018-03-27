Per ora la configurazione migliore risulta essere la LeNet arrivando a picchi di accuratezza
anche fino al 90%

my_number --> classificazione 1 e 2 con sfondo bianco funziona bene con sfondo colorato un po
meno (vedi result 54)

Provato ad aggiungere nuovi immagini ad 1 e 2 (result 55)

In number provare la classificazione con i numeri da 1 a 5

Per ora il migliore risultato l'ho ottenuto con il 75 (background_black_200x200)
con una accuracy del 98% il problema è che caricandola poi nell'applicazione non funziona 
come deovrebbe

Provare poi ad addestrare la rete con le immagini a sfondo bianco 

Provatre con tutte le immagini scattata diverse sono circa 3216

Provare con le dimensioni 250 altezza e 200 di larghezza --> model82 senza trasformazioni

Provare con alcune trasformazioni --> model83 un paio di trasformazioni
Provare model82 con VGG16 ImagePreprocessing