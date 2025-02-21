import * as React from "react";
import { useEffect, useState } from "react";
import { FrameNavigationProp } from "react-nativescript-navigation";
import { MainStackParamList } from "../NavigationParamList";

type HomeScreenProps = {
  navigation: FrameNavigationProp<MainStackParamList, "OptiFeux">;
};

export function HomeScreen({ navigation }: HomeScreenProps) {
  const parkingStats = require("../assets/parkingData.json"); // ✅ Charge les données localement
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredZones, setFilteredZones] = useState(parkingStats.zones);
  const [showSearchBar, setShowSearchBar] = useState(false); // ✅ Contrôle l'affichage du champ de recherche

  // ✅ Fonction pour filtrer les zones en fonction du texte saisi
  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (query.trim() === "") {
      setFilteredZones(parkingStats.zones); // ✅ Si vide, afficher toutes les zones
    } else {
      const filtered = parkingStats.zones.filter((zone: any) =>
        zone.name.toLowerCase().includes(query.toLowerCase())
      );
      setFilteredZones(filtered);
    }
  };

  return (
    <gridLayout rows="auto, *, auto">
      {/* ✅ Fixe le header en haut, en dehors du scrollView */}
      <stackLayout row="0" className="gradient-header p-6">
        <gridLayout rows="auto, auto" columns="*, auto" className="mb-4">
          <label
            text="Trouvez votre place facilement"
            className="text-white text-center t-1"
          />
        </gridLayout>

        {/* ✅ Barre de recherche affichée dynamiquement */}
        {showSearchBar && (
          <textField
            className="bg-white text-black p-2 rounded-lg mt-4"
            placeholder="Rechercher une zone..."
            text={searchQuery}
            onTextChange={(event) => handleSearch(event.value)}
          />
        )}
      </stackLayout>

      {/* ✅ Ajoute un scroll uniquement pour le contenu */}
      <scrollView row="1" className="page">
        <stackLayout>
          {/* ✅ Garde les statistiques en-dessous du header */}
          <gridLayout columns="*, *" rows="auto" className="mx-4 mb-6">
            <stackLayout col="0" className="stat-card m-2 p-4">
              <label text="🚗" className="text-2xl mb-2" />
              <label
                text={parkingStats.globalStats.availableSpots.toString()}
                className="text-2xl font-bold text-blue-600"
              />
              <label text="Places libres" className="text-sm text-gray-600" />
            </stackLayout>
            <stackLayout col="1" className="stat-card m-2 p-4">
              <label text="⚡" className="text-2xl mb-2" />
              <label
                text={parkingStats.globalStats.chargingStations.toString()}
                className="text-2xl font-bold text-green-600"
              />
              <label
                text="Bornes disponibles"
                className="text-sm text-gray-600"
              />
            </stackLayout>
          </gridLayout>

          {/* ✅ Affichage dynamique des zones (filtrées ou toutes) */}
          <stackLayout className="px-4">
            <label
              text="Zones de stationnement"
              className="text-xl font-bold text-gray-800 mb-4"
            />
            {filteredZones.length > 0 ? (
              filteredZones.map((zone: any) => (
                <gridLayout
                  key={zone.id}
                  columns="*, auto"
                  className="card p-4 mb-3"
                  onTap={() =>
                    navigation.navigate("Zone Details", { zoneId: zone.id })
                  }
                >
                  <stackLayout col="0">
                    <label
                      text={zone.name}
                      className="text-lg font-bold text-gray-800"
                    />
                    <gridLayout columns="auto, auto" className="mt-2">
                      <label
                        col="0"
                        text={`${zone.available}/${zone.total}`}
                        className="text-blue-600 font-bold"
                      />
                      <label col="1" text=" places" className="text-gray-600" />
                    </gridLayout>
                  </stackLayout>
                  <stackLayout col="1" verticalAlignment="center">
                    <label
                      text={zone.available > 5 ? "✓ Disponible" : "× Limité"}
                      className={`text-sm text-white px-4 py-2 rounded-full ${
                        zone.available > 5 ? "bg-green-500" : "bg-red-500"
                      }`}
                      style={{ textAlign: "center", minWidth: 80 }}
                    />
                  </stackLayout>
                </gridLayout>
              ))
            ) : (
              <label
                text="Aucune zone trouvée."
                className="text-center text-lg text-gray-500"
              />
            )}
          </stackLayout>
        </stackLayout>
      </scrollView>

      {/* ✅ Fixe la barre de navigation en bas */}
      <gridLayout
        row="2"
        columns="*, *, *"
        className="bg-white border-t border-gray-200 gridLayout"
      >
        <stackLayout
          col="0"
          className="stackLayout"
          onTap={() => setShowSearchBar(!showSearchBar)}
        >
          <label text="🔍" className="text-2xl" />
          <label text="Recherche" className="text-xs text-gray-600" />
        </stackLayout>
        <stackLayout
          col="2"
          className="stackLayout"
          onTap={() => navigation.navigate("Carte")}
        >
          <label text="🗺️" className="text-2xl" />
          <label text="Carte" className="text-xs text-gray-600" />
        </stackLayout>
      </gridLayout>
    </gridLayout>
  );
}
