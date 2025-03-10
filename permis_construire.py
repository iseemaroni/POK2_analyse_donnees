import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os

from sklearn.linear_model import LinearRegression

import geopandas as gpd



# Charger les données

#Non résidentiel
data1 = pd.read_csv(
    '/Users/iseemaroni/Documents/Documents/Centrale Marseille/3A/POK & MON/POK 3/Liste-des-autorisations-durbanisme-creant-des-locaux-non-residentiels.2024-12 (1).csv', 
    sep=';', 
    header=1
)

#Logements
data2 = pd.read_csv(
    '/Users/iseemaroni/Documents/Documents/Centrale Marseille/3A/POK & MON/POK 3/Liste-des-autorisations-durbanisme-creant-des-logements.2024-12.csv', 
    sep=';', 
    header=1
)

# Nettoyage des colonnes
data1.columns = data1.columns.str.replace('"', '').str.strip()
data2.columns = data2.columns.str.replace('"', '').str.strip()

# Load the data
vente1 = pd.read_csv(
    "/Users/iseemaroni/Documents/Documents/Centrale Marseille/3A/POK & MON/POK 3/Estimated_Annual_Sales_of_Hard_Hats_in _France_2.csv", 
    sep=';', 
    header=0)  # Adjust separator if needed



##### Liste des autorisations d'urbanisme créant des locaux non résidentiels

def PCNonResidentiel():
    # Charger les données
    global data1

    # Convertir les dates et extraire l'année
    data1['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data1['DATE_REELLE_AUTORISATION'], errors='coerce')
    data1['ANNEE_AUTORISATION'] = data1['DATE_REELLE_AUTORISATION'].dt.year

    # Compter le nombre de permis par année
    permis_par_annee = data1.groupby('ANNEE_AUTORISATION').size().reset_index(name='Nombre de Permis')

    # Tracer le graphique
    sns.barplot(x='ANNEE_AUTORISATION', y='Nombre de Permis', data=permis_par_annee, palette='viridis')
    plt.title('Nombre de Permis de Construire non résidentiels par Année')
    plt.xlabel('Année')
    plt.ylabel('Nombre de Permis')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def PCNonResidentiel2():

    # Charger les données
    global data1

    # Convertir les dates et extraire l'année
    data1['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data1['DATE_REELLE_AUTORISATION'], errors='coerce')
    data1['ANNEE_AUTORISATION'] = data1['DATE_REELLE_AUTORISATION'].dt.year

    # Compter le nombre de permis par année
    permis_par_annee = data1.groupby('ANNEE_AUTORISATION').size().reset_index(name='Nombre de Permis')

    # Normaliser les hauteurs pour correspondre à une échelle de couleurs inversée
    norm = plt.Normalize(permis_par_annee['Nombre de Permis'].min(), permis_par_annee['Nombre de Permis'].max())
    sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)  # Utilisation de la palette inversée

    # Ajouter une colonne avec les couleurs basées sur la hauteur
    permis_par_annee['Couleur'] = permis_par_annee['Nombre de Permis'].apply(lambda x: sm.to_rgba(x))

    # Tracer le graphique
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        permis_par_annee['ANNEE_AUTORISATION'], 
        permis_par_annee['Nombre de Permis'], 
        color=permis_par_annee['Couleur']
    )

    # Ajouter une barre de couleurs pour la légende
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Nombre de Permis')

    # Ajouter des détails au graphique
    plt.title('Nombre de Permis de Construire non résidentiels par Année', fontsize=16)
    plt.xlabel('Année', fontsize=14)
    plt.ylabel('Nombre de Permis', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# PCNonResidentiel2()




##### Liste des autorisations d'urbanisme créant des logements


def PCResidentiel():
    # Charger les données
    global data2

    # Convertir les dates et extraire l'année
    data2['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data2['DATE_REELLE_AUTORISATION'], errors='coerce')
    data2['ANNEE_AUTORISATION'] = data2['DATE_REELLE_AUTORISATION'].dt.year

    # Compter le nombre de permis par année
    permis_par_annee = data2.groupby('ANNEE_AUTORISATION').size().reset_index(name='Nombre de Permis')

    # Tracer le graphique
    sns.barplot(x='ANNEE_AUTORISATION', y='Nombre de Permis', data=permis_par_annee, palette='viridis')
    plt.title('Nombre de Permis de Construire de logements par Année')
    plt.xlabel('Année')
    plt.ylabel('Nombre de Permis')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# PCResidentiel()



def superpositionHistograms():

    global data1
    global data2

    # Convertir les dates en datetime et extraire l'année
    data1['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data1['DATE_REELLE_AUTORISATION'], errors='coerce')
    data1['ANNEE_AUTORISATION'] = data1['DATE_REELLE_AUTORISATION'].dt.year

    data2['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data2['DATE_REELLE_AUTORISATION'], errors='coerce')
    data2['ANNEE_AUTORISATION'] = data2['DATE_REELLE_AUTORISATION'].dt.year

    plt.figure(figsize=(10, 6))

    # Palette personnalisée
    palette = ['brown', 'orange']


    # Comptage des occurrences pour data1
    frequences_data1 = data1['ANNEE_AUTORISATION'].value_counts().sort_index()

    # Comptage des occurrences pour data2
    frequences_data2 = data2['ANNEE_AUTORISATION'].value_counts().sort_index()

    # Affichage du récapitulatif
    print("Occurrences pour les locaux non résidentiels :")
    for annee, count in frequences_data1.items():
        print(f"{annee} : {count} occurrences")

    print("\nOccurrences pour les logements :")
    for annee, count in frequences_data2.items():
        print(f"{annee} : {count} occurrences")
    

    # Histogrammes
    sns.histplot(data1['ANNEE_AUTORISATION'], label='Locaux non résidentiels', color=palette[0], kde=False, binwidth=1, binrange=(2013, 2025))
    sns.histplot(data2['ANNEE_AUTORISATION'], label='Logements', color=palette[1], kde=False, binwidth=1, binrange=(2013, 2025))

    # Définir les ticks manuels sur l'axe des x
    annees = sorted(set(data1['ANNEE_AUTORISATION']).union(set(data2['ANNEE_AUTORISATION'])))
    plt.xticks(ticks=annees, labels=annees, rotation=45, fontsize=10)  # Rotation si les années sont nombreuses


    # Affichage de la légende (automatique)
    plt.legend(title="Catégories")

    # Ajouter des titres et autres détails
    plt.title("Superposition des distributions des autorisations d'urbanisme par année", fontsize=16)
    plt.xlabel("Année", fontsize=14)
    plt.ylabel("Nombre d'autorisations", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()

# superpositionHistograms()



def addition():

    global data1
    global data2

    # Convertir les dates en datetime et extraire l'année
    data1['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data1['DATE_REELLE_AUTORISATION'], errors='coerce')
    data1['ANNEE_AUTORISATION'] = data1['DATE_REELLE_AUTORISATION'].dt.year

    data2['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data2['DATE_REELLE_AUTORISATION'], errors='coerce')
    data2['ANNEE_AUTORISATION'] = data2['DATE_REELLE_AUTORISATION'].dt.year

    # Combiner les deux séries en sommant les occurrences par année
    combined_data = (
        pd.concat([data1['ANNEE_AUTORISATION'], data2['ANNEE_AUTORISATION']])
        .value_counts()
        .sort_index()
    )

    # Tracer l'histogramme avec les données combinées
    plt.figure(figsize=(12, 6))
    sns.barplot(x=combined_data.index, y=combined_data.values, color='brown')

    # Ajouter des détails au graphique
    plt.title("Évolution annuelle du nombre total de permis de construire", fontsize=16)
    plt.xlabel("Année", fontsize=14)
    plt.ylabel("Nombre d'autorisations de construire", fontsize=14)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # return(combined_data.index, combined_data.values)

# addition()


def superpositionDensities():

    global data1  # Indique que vous utilisez la variable globale
    global data2

    # Convertir les dates et extraire les mois
    data1['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data1['DATE_REELLE_AUTORISATION'], errors='coerce')
    data2['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data2['DATE_REELLE_AUTORISATION'], errors='coerce')

    data1 = data1.dropna(subset=['DATE_REELLE_AUTORISATION'])
    data2 = data2.dropna(subset=['DATE_REELLE_AUTORISATION'])

    # Ajouter colonne MOIS et comptage par mois
    data1['MOIS'] = data1['DATE_REELLE_AUTORISATION'].dt.to_period('M').dt.to_timestamp()
    data2['MOIS'] = data2['DATE_REELLE_AUTORISATION'].dt.to_period('M').dt.to_timestamp()

    # Grouper par mois
    permis_data1 = data1.groupby('MOIS').size().reset_index(name='Nombre de Permis')
    permis_data2 = data2.groupby('MOIS').size().reset_index(name='Nombre de Permis')

    # Debug : Afficher les données transformées
    print("Locaux non résidentiels :\n", permis_data1.head())
    print("Logements :\n", permis_data2.head())

    # Tracer les densités pondérées
    plt.figure(figsize=(12, 8))

    sns.kdeplot(
        data=permis_data1, 
        x='MOIS', 
        weights='Nombre de Permis', 
        label="Locaux non résidentiels", 
        fill=True, 
        alpha=0.5, 
        color="blue", 
        bw_adjust=0.5
    )
    sns.kdeplot(
        data=permis_data2, 
        x='MOIS', 
        weights='Nombre de Permis', 
        label="Logements", 
        fill=True, 
        alpha=0.5, 
        color="orange", 
        bw_adjust=0.5
    )

    # Ajuster les axes
    plt.title("Densités des autorisations d'urbanisme par mois", fontsize=16)
    plt.xlabel("Année", fontsize=14)
    plt.ylabel("Densité pondérée par nombre de permis", fontsize=14)

    # Réglage des ticks de l'axe X (affichage par année seulement)
    plt.xticks(
        ticks=pd.date_range(start=permis_data1['MOIS'].min(), end=permis_data1['MOIS'].max(), freq='YS'),
        labels=pd.date_range(start=permis_data1['MOIS'].min(), end=permis_data1['MOIS'].max(), freq='YS').year,
        rotation=45
    )

    plt.legend(title="Catégories")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# superpositionDensities()

# superposition Densities Scaled To Total Area()

def superpositionDensities2():
    global data1
    global data2

    # Convertir les dates et extraire les mois
    data1['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data1['DATE_REELLE_AUTORISATION'], errors='coerce')
    data2['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data2['DATE_REELLE_AUTORISATION'], errors='coerce')

    data1 = data1.dropna(subset=['DATE_REELLE_AUTORISATION'])
    data2 = data2.dropna(subset=['DATE_REELLE_AUTORISATION'])

    # Extraire les mois
    data1['MOIS'] = data1['DATE_REELLE_AUTORISATION'].dt.to_period('M').dt.to_timestamp()
    data2['MOIS'] = data2['DATE_REELLE_AUTORISATION'].dt.to_period('M').dt.to_timestamp()

    # Calculer les totaux
    total_permis1 = len(data1)
    total_permis2 = len(data2)

    # Calculer les densités
    from scipy.stats import gaussian_kde
    kde1 = gaussian_kde(data1['MOIS'].map(pd.Timestamp.toordinal), bw_method=0.15)
    kde2 = gaussian_kde(data2['MOIS'].map(pd.Timestamp.toordinal), bw_method=0.15)

    # Générer une plage de dates pour l'axe X
    x_range = pd.date_range(
        start=min(data1['MOIS'].min(), data2['MOIS'].min()),
        end=max(data1['MOIS'].max(), data2['MOIS'].max()),
        freq='M'
    )
    x_ordinal = x_range.map(pd.Timestamp.toordinal)

    # Calculer les densités ajustées
    density1 = kde1(x_ordinal) * total_permis1
    density2 = kde2(x_ordinal) * total_permis2

    print("Densité ajustée pour Locaux non résidentiels :", density1[:5])
    print("Densité ajustée pour Logements :", density2[:5])

    # Tracer les courbes
    plt.figure(figsize=(12, 8))

    plt.fill_between(x_range, density1, alpha=0.5, label="Locaux non résidentiels", color="blue",)
    plt.fill_between(x_range, density2, alpha=0.5, label="Logements", color="orange")

    # Ajuster les axes
    plt.title("Densités des autorisations d'urbanisme (proportionnelles aux totaux respectifs)", fontsize=16)
    plt.xlabel("Année", fontsize=14)
    plt.ylabel("Densité pondérée par nombre de permis", fontsize=14)

    # Réglage des ticks de l'axe X (affichage par année seulement)
    plt.xticks(
        ticks=pd.date_range(start=x_range.min(), end=x_range.max(), freq='YS'),
        labels=pd.date_range(start=x_range.min(), end=x_range.max(), freq='YS').year,
        rotation=45
    )

    plt.xlim(pd.Timestamp('2013-01-01'), pd.Timestamp('2024-12-31'))
    plt.legend(title="Catégories")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()




# superpositionDensities2()
    
def separateDensities():
    global data1
    global data2

    # Convertir les dates et extraire les mois
    data1['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data1['DATE_REELLE_AUTORISATION'], errors='coerce')
    data2['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data2['DATE_REELLE_AUTORISATION'], errors='coerce')

    data1 = data1.dropna(subset=['DATE_REELLE_AUTORISATION'])
    data2 = data2.dropna(subset=['DATE_REELLE_AUTORISATION'])

    # Extraire les mois
    data1['MOIS'] = data1['DATE_REELLE_AUTORISATION'].dt.to_period('M').dt.to_timestamp()
    data2['MOIS'] = data2['DATE_REELLE_AUTORISATION'].dt.to_period('M').dt.to_timestamp()

    # Calculer les totaux
    total_permis1 = len(data1)
    total_permis2 = len(data2)
    total_permis = total_permis1 + total_permis2

    # Calculer les densités avec un lissage contrôlé
    from scipy.stats import gaussian_kde
    kde1 = gaussian_kde(data1['MOIS'].map(pd.Timestamp.toordinal), bw_method=0.2)
    kde2 = gaussian_kde(data2['MOIS'].map(pd.Timestamp.toordinal), bw_method=0.2)

    # Générer une plage de dates pour l'axe X
    x_range = pd.date_range(
        start=min(data1['MOIS'].min(), data2['MOIS'].min()),
        end=max(data1['MOIS'].max(), data2['MOIS'].max()),
        freq='M'
    )
    x_ordinal = x_range.map(pd.Timestamp.toordinal)

    # Calculer les densités pondérées
    density1 = kde1(x_ordinal) * (total_permis1 / total_permis)
    density2 = kde2(x_ordinal) * (total_permis2 / total_permis)

    # Créer les sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Graphe 1 : Locaux non résidentiels
    axes[0].fill_between(x_range, density1, alpha=0.5, label="Locaux non résidentiels", color="blue")
    axes[0].set_title("Densité des locaux non résidentiels (proportion de l'ensemble)", fontsize=16)
    axes[0].set_xlabel("Année", fontsize=14)
    axes[0].set_ylabel("Proportion des permis", fontsize=14)
    axes[0].legend(title="Catégories")
    axes[0].grid(alpha=0.3)

    # Graphe 2 : Logements
    axes[1].fill_between(x_range, density2, alpha=0.5, label="Logements", color="orange")
    axes[1].set_title("Densité des logements (proportion de l'ensemble)", fontsize=16)
    axes[1].set_xlabel("Année", fontsize=14)
    axes[1].legend(title="Catégories")
    axes[1].grid(alpha=0.3)

    # Réglage des ticks de l'axe X (affichage par année seulement)
    ticks = pd.date_range(start=x_range.min(), end=x_range.max(), freq='YS')
    labels = ticks.year
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlim(pd.Timestamp('2013-01-01'), pd.Timestamp('2024-12-31'))

    plt.tight_layout()
    plt.show()

# Appeler la fonction
# separateDensities()


def separateDensities2():
    global data1, data2

    # Assurez-vous que les données sont bien chargées et traitées
    data1['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data1['DATE_REELLE_AUTORISATION'], errors='coerce')
    data2['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data2['DATE_REELLE_AUTORISATION'], errors='coerce')

    data1 = data1.dropna(subset=['DATE_REELLE_AUTORISATION'])
    data2 = data2.dropna(subset=['DATE_REELLE_AUTORISATION'])

    # Ajouter la colonne mois pour grouper par mois
    data1['MOIS'] = data1['DATE_REELLE_AUTORISATION'].dt.to_period('M').dt.to_timestamp()
    data2['MOIS'] = data2['DATE_REELLE_AUTORISATION'].dt.to_period('M').dt.to_timestamp()

    # Regrouper par mois et compter les permis
    permis_data1 = data1.groupby('MOIS').size().reset_index(name='Nombre de Permis')
    permis_data2 = data2.groupby('MOIS').size().reset_index(name='Nombre de Permis')

    # Calculer la distribution totale en combinant les deux jeux de données
    permis_total = pd.concat([permis_data1, permis_data2]).groupby('MOIS').sum().reset_index()

    # Ajouter des colonnes de proportions basées sur le total combiné
    total_permits = permis_total['Nombre de Permis'].sum()
    permis_data1['Proportion'] = permis_data1['Nombre de Permis'] / total_permits
    permis_data2['Proportion'] = permis_data2['Nombre de Permis'] / total_permits
    permis_total['Proportion'] = permis_total['Nombre de Permis'] / total_permits

    # Création de la figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # Densité de "Locaux non résidentiels" et "Logements"
    sns.kdeplot(
        data=permis_data1,
        x='MOIS',
        weights='Proportion',
        fill=True,
        alpha=0.5,
        color='blue',
        label='Locaux non résidentiels',
        ax=axes[0]
    )

    sns.kdeplot(
        data=permis_data2,
        x='MOIS',
        weights='Proportion',
        fill=True,
        alpha=0.5,
        color='orange',
        label='Logements',
        ax=axes[0]
    )

    # Calcul des densités KDE avec gaussian_kde
    kde1 = gaussian_kde(permis_data1['MOIS'].astype(np.int64), weights=permis_data1['Proportion'])
    kde2 = gaussian_kde(permis_data2['MOIS'].astype(np.int64), weights=permis_data2['Proportion'])

    # Création d'un axe temporel commun
    common_time_range = np.linspace(permis_data1['MOIS'].min().astype(np.int64), permis_data1['MOIS'].max().astype(np.int64), 1000)

    # Calcul des densités pour chaque catégorie
    density1 = kde1(common_time_range)
    density2 = kde2(common_time_range)

    # Densité combinée (somme des deux densités)
    total_density = density1 + density2

    # Tracer la densité combinée en gris
    axes[0].fill_between(pd.to_datetime(common_time_range), total_density, color='gray', alpha=0.3, label='Densité combinée')

    axes[0].set_title("Distribution: Locaux non résidentiels", fontsize=14)
    axes[0].set_xlabel("Année", fontsize=12)
    axes[0].set_ylabel("Proportion", fontsize=12)

    # Tracer sur le graphique de "Logements"
    sns.kdeplot(
        data=permis_data2,
        x='MOIS',
        weights='Proportion',
        fill=True,
        alpha=0.5,
        color='orange',
        label='Logements',
        ax=axes[1]
    )

    sns.kdeplot(
        data=permis_data1,
        x='MOIS',
        weights='Proportion',
        fill=True,
        alpha=0.5,
        color='blue',
        label='Locaux non résidentiels',
        ax=axes[1]
    )

    # Tracer la densité combinée en gris
    axes[1].fill_between(pd.to_datetime(common_time_range), total_density, color='gray', alpha=0.3, label='Densité combinée')

    axes[1].set_title("Distribution: Logements", fontsize=14)
    axes[1].set_xlabel("Année", fontsize=12)

    # Ajuster les limites de l'axe x pour les deux graphiques
    for ax in axes:
        ax.set_xlim([pd.Timestamp('2013-01-01'), pd.Timestamp('2024-12-31')])
        ax.set_xticks(pd.date_range(start='2013-01-01', end='2024-12-31', freq='YS'))
        ax.set_xticklabels(pd.date_range(start='2013-01-01', end='2024-12-31', freq='YS').year, rotation=45)

    plt.tight_layout()
    plt.show()


# separateDensities2()


file_path = "/Users/iseemaroni/Documents/Documents/Centrale Marseille/3A/POK & MON/POK 3/Estimated_Annual_Sales_of_Hard_Hats_in _France_2.csv"
print(os.path.exists(file_path))  # Doit afficher True si le fichier existe

def VentesTendance():

    # Load the data
    global vente1

    # Plot the data
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=vente1, x="Year", y="Estimated Sales (Million Units)", marker="o", color="brown")

    # Customize the plot
    plt.title("Estimation des ventes de casques de sécurité en France par ChatGPT", fontsize=14)
    plt.xlabel("Année", fontsize=12)
    plt.ylabel("Ventes (millions)", fontsize=12)
    plt.xticks(rotation=0)
    plt.xticks(vente1["Year"])
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()

# VentesTendance()
    

# x,y = addition(combined_data.index, combined_data.values)

def correlation():
    global data1
    global data2
    global vente1

    # Convertir les dates en datetime et extraire l'année
    data1['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data1['DATE_REELLE_AUTORISATION'], errors='coerce')
    data1['ANNEE_AUTORISATION'] = data1['DATE_REELLE_AUTORISATION'].dt.year

    data2['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data2['DATE_REELLE_AUTORISATION'], errors='coerce')
    data2['ANNEE_AUTORISATION'] = data2['DATE_REELLE_AUTORISATION'].dt.year

    # Combiner les deux séries en sommant les occurrences par année
    combined_data = (
        pd.concat([data1['ANNEE_AUTORISATION'], data2['ANNEE_AUTORISATION']])
        .value_counts()
        .sort_index()
    )
    sales = vente1["Estimated Sales (Million Units)"].tolist()

    #print(combined_data.values)
    #print(sales)

    # Tracer l'histogramme avec les données combinées
    #plt.figure(figsize=(12, 6))
    #sns.scatterplot(x=sales, y=combined_data.values)

    # Ajouter des labels
   # plt.xlabel("Nombre total de permis de construire")
    #plt.ylabel("Ventes estimées de casques (millions)")
    #plt.title("Corrélation entre permis de co nstruire et ventes de casques")
    #plt.tight_layout()
    #plt.show()

    # Reshape pour scikit-learn (X doit être 2D)
    X = np.array(combined_data.values).reshape(-1, 1)
    y = np.array(sales)

    # Création et entraînement du modèle
    model = LinearRegression()
    model.fit(X, y)

    # Prédictions
    y_pred = model.predict(X)

    # Tracer la droite de régression
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=sales, y=combined_data.values, label="Données réelles")
    plt.plot(sales, y_pred, color="red", label="Régression linéaire")

    # Récupération des coefficients
    coef = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    # Ajouter les coefficients sur le graphe
    equation = f"y = {coef:.4f}x + {intercept:.4f}\nR² = {r2:.4f}"
    plt.annotate(equation, xy=(0.70, 0.10), xycoords="axes fraction", fontsize=12, color="darkred")

    # Ajouter des labels
    plt.xlabel("Nombre total de permis de construire")
    plt.ylabel("Ventes estimées de casques (millions)")
    plt.title("Corrélation entre permis de construire et ventes de casques")
    plt.legend()
    plt.show()


# correlation()
    

def carte_chaleur_par_annee():
    global data1, data2

    # Convertir les dates en datetime et extraire l'année
    data1['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data1['DATE_REELLE_AUTORISATION'], errors='coerce')
    data1['ANNEE_AUTORISATION'] = data1['DATE_REELLE_AUTORISATION'].dt.year

    data2['DATE_REELLE_AUTORISATION'] = pd.to_datetime(data2['DATE_REELLE_AUTORISATION'], errors='coerce')
    data2['ANNEE_AUTORISATION'] = data2['DATE_REELLE_AUTORISATION'].dt.year

    # Combiner les deux séries en sommant les occurrences par année
    combined_data = (
        pd.concat([data1['ANNEE_AUTORISATION'], data2['ANNEE_AUTORISATION']])
        .value_counts()
        .sort_index()
    )

    # Transformer en DataFrame 2D (Heatmap attend une matrice)
    df_heatmap = pd.DataFrame(combined_data).T  # Convertir en ligne unique

    # Tracer la carte de chaleur
    plt.figure(figsize=(12, 3))  # Adapter la taille pour une heatmap horizontale
    sns.heatmap(df_heatmap, cmap="Reds", annot=False, fmt="d", linewidths=0.5, cbar=True,vmin=100000)

    # Ajustements graphiques
    plt.title("Carte de chaleur des permis de construire par année", fontsize=14)
    plt.xlabel("Année", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks([])  # Supprimer les labels inutiles sur Y (ligne unique)

    plt.show()

# Appeler la fonction pour afficher la heatmap
# carte_chaleur_par_annee()
    

def carte_par_departement():
    # Charger les données des permis de construire
    global data2

    # Compter le nombre de permis par département
    permis_par_dept = (data2['DEP_CODE'].astype(str).str.zfill(2)).value_counts().reset_index()
    permis_par_dept.columns = ['DEP_CODE', 'Nombre de permis']

    # Charger la carte des départements français
    url_geojson = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    france_map = gpd.read_file(url_geojson)

    # Assurer la compatibilité des codes départementaux
    france_map['code'] = france_map['code'].astype(str)
    
    # Fusionner avec les données de permis
    france_map = france_map.merge(permis_par_dept, left_on='code', right_on='DEP_CODE', how='left')
    france_map['Nombre de permis'] = france_map['Nombre de permis'].fillna(0)  # Remplacer les NaN par 0

    # Tracer la carte
    fig, ax = plt.subplots(figsize=(10, 8))
    france_map.plot(column='Nombre de permis', cmap='Reds', linewidth=0.5, edgecolor='black', legend=True, ax=ax, vmin=500)

    # Ajustements du graphique
    ax.set_title("Densité des permis de construire de logements par département en France de 2013 à 2024 ", fontsize=14)
    ax.axis("off")  # Supprime les axes

    # Afficher la carte
    plt.show()

# Exemple d'utilisation
carte_par_departement()


def carte_par_departement2(valeur_min=0):
    global data1

    # Compter le nombre de permis par département
    permis_par_dept = (data1['DEP_CODE'].astype(str).str.zfill(2)).value_counts().reset_index()
    permis_par_dept.columns = ['DEP_CODE', 'Nombre de permis']

    # Charger la carte des départements français
    url_geojson = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    france_map = gpd.read_file(url_geojson)

    # Assurer la compatibilité des codes départementaux
    # Convertir en string avec zéro en préfixe pour les numéros à 1 chiffre
    france_map['code'] = france_map['code'].astype(str)
    
    # Fusionner avec les données de permis
    france_map = france_map.merge(permis_par_dept, left_on='code', right_on='DEP_CODE', how='left')
    france_map['Nombre de permis'] = france_map['Nombre de permis'].fillna(0)  # Remplacer les NaN par 0

    # Définir la plage de valeurs pour l'échelle des couleurs
    vmax = france_map['Nombre de permis'].max()
    vmin = max(valeur_min, france_map['Nombre de permis'].min())  # Assurer une valeur minimale cohérente

    # Tracer la carte
    fig, ax = plt.subplots(figsize=(10, 8))
    france_map.plot(column='Nombre de permis', cmap='Reds', linewidth=0.5, edgecolor='black', 
                    legend=True, ax=ax, vmin=vmin, vmax=vmax)

    # Ajouter les valeurs des départements sur la carte
    for _, row in france_map.iterrows():
        if row['Nombre de permis'] > 0:  # Éviter d'afficher les valeurs nulles
            plt.text(row.geometry.centroid.x, row.geometry.centroid.y, 
                     int(row['Nombre de permis']), fontsize=8, ha='center', color='black')

    # Ajustements du graphique
    ax.set_title("Densité des permis de construire non résidentiel par département en France de 2013 à 2024", fontsize=14)
    ax.axis("off")  # Supprime les axes

    # Afficher la carte
    plt.show()

# Exemple d'utilisation avec un minimum de 10 pour l'échelle des couleurs
# carte_par_departement2(valeur_min=10)


def carte_par_departement_par_annee(annee=None):
    global data1

    # Extraire l'année
    data1['ANNEE_AUTORISATION'] = pd.to_datetime(data1['DATE_REELLE_AUTORISATION'], errors='coerce').dt.year

    # Filtrer par année si précisé
    if annee:
        data_filtered = data1[data1['ANNEE_AUTORISATION'] == annee]
    else:
        data_filtered = data1

    # Compter les permis par département
    permis_par_dept = (data1['DEP_CODE'].astype(str).str.zfill(2)).value_counts().reset_index()
    permis_par_dept.columns = ['DEP_CODE', 'Nombre de permis']

    # Charger la carte des départements
    url_geojson = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    france_map = gpd.read_file(url_geojson)

    # Assurer la compatibilité des codes départementaux
    france_map['code'] = france_map['code'].astype(str)
    
    # Fusionner avec les données de permis
    france_map = france_map.merge(permis_par_dept, left_on='code', right_on='DEP_CODE', how='left')
    france_map['Nombre de permis'] = france_map['Nombre de permis'].fillna(0)

    # Tracer la carte
    fig, ax = plt.subplots(figsize=(10, 8))
    france_map.plot(column='Nombre de permis', cmap='Reds', linewidth=0.5, edgecolor='black', legend=True, ax=ax)

    # Ajustements
    titre = f"Densité des permis de construire par département en {annee}" if annee else "Densité des permis de construire par département"
    ax.set_title(titre, fontsize=14)
    ax.axis("off")

    plt.show()

# Exemple : afficher la carte pour 2020
# carte_par_departement_par_annee(annee=2016)