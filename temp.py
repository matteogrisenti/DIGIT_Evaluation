import torch

# Inserisci qui il percorso corretto del file .ckpt del tuo decoder
checkpoint_path = "outputs_sparsh/checkpoints/forcefield_decoder/digit_t1_forcefield_dino_vitbase_bg/checkpoints/last.ckpt" 

print("Caricamento del checkpoint in corso...")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# 1. Verifichiamo la presenza della macro-chiave "model" scoperto nel dump precedente
if "model" not in checkpoint:
    print("\n[ERRORE] La chiave 'model' non esiste nel file! Ecco cosa c'è invece:")
    print(list(checkpoint.keys()))
    exit()

# Estraiamo il vero dizionario dei pesi racchiuso dentro "model"
real_weights_dict = checkpoint["model"]

print("\n" + "="*70)
print("   VERIFICA CHIRURGICA DEI PESI INTERNI AL DECODER (DENTRO 'model')")
print("="*70)
print(f"Numero totale di tensori neurali scoperti: {len(real_weights_dict.keys())}\n")

# Contatori per verificare se i blocchi dell'architettura esistono davvero nel file
found_layers = {
    "norm": [],
    "reassembles": [],
    "fusions": [],
    "head": []
}

# Analizziamo le chiavi interne per catalogarle
for key in sorted(real_weights_dict.keys()):
    # Puliamo momentaneamente i prefissi per fare il controllo del nome nativo dello strato
    clean_key = key.replace("model_task.", "").replace("module.", "").replace("decoder.", "")
    
    if "norm" in clean_key:
        found_layers["norm"].append((key, real_weights_dict[key]))
    elif "reassembles" in clean_key:
        found_layers["reassembles"].append((key, real_weights_dict[key]))
    elif "fusions" in clean_key:
        found_layers["fusions"].append((key, real_weights_dict[key]))
    elif "head" in clean_key:
        found_layers["head"].append((key, real_weights_dict[key]))

# 2. STAMPA DEI RISULTATI DELLA VERIFICA
print(f"-> Strati 'reassembles' trovati nel file: {len(found_layers['reassembles'])}")
print(f"-> Strati 'fusions' trovati nel file: {len(found_layers['fusions'])}")
print(f"-> Strati 'head' (Testa predittiva) trovati nel file: {len(found_layers['head'])}")
print(f"-> Strati di normalizzazione 'norm' trovati nel file: {len(found_layers['norm'])}\n")

print("-"*70)
print(" DIMOSTRAZIONE VISIVA: ESTRATTO DEI VERI VALORI NUMERICI DEI PESI")
print("-"*70)

# Prendiamo un esempio per ogni blocco per dimostrare che i dati numerici sono presenti
# e mostriamo come PyTorch li chiama nel file originale rispetto al decoder locale
for layer_type, layers in found_layers.items():
    if len(layers) > 0:
        target_key, tensor_data = layers[0] # Prende il primo peso disponibile di quel tipo
        
        # Estrarre un pezzetto di numeri reali dal tensore per vederli fisicamente
        flat_data = tensor_data.flatten().numpy()
        sample_values = [f"{x:.4f}" for x in flat_data[:4]] # Primi 4 numeri reali
        
        print(f"Strato Tipo [{layer_type.upper()}]:")
        print(f"   • Chiave originale nel file .ckpt : '{target_key}'")
        print(f"   • Forma geometrica del blocco     : {list(tensor_data.shape)}")
        print(f"   • Campione primi 4 pesi reali     : {sample_values}")
        print()

print("="*70)
print("Fine del test di ispezione interna.")