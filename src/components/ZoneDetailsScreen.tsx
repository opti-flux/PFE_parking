import * as React from "react";
import { useEffect, useState } from "react";
import { FrameNavigationProp } from "react-nativescript-navigation";
import { MainStackParamList } from "../NavigationParamList";

type ZoneDetailsProps = {
  navigation: FrameNavigationProp<MainStackParamList, "Zone Details">;
  route: { params: { zoneId: number } };
};

export function ZoneDetailsScreen({ route }: ZoneDetailsProps) {
  const { zoneId } = route.params;
  const parkingData = require("../assets/parkingData.json"); // ✅ Charge les données localement
  const zoneData = parkingData.zones.find((zone: any) => zone.id === zoneId);

  if (!zoneData) {
    return <label text="Chargement..." className="text-center text-lg" />;
  }

  // Construire l'URL de la carte avec les coordonnées de la zone
  const mapUrl = `~/assets/map.html?lat=${zoneData.coordinates.latitude}&lng=${
    zoneData.coordinates.longitude
  }&name=${encodeURIComponent(zoneData.name)}`;

  return (
    <gridLayout rows="auto, *, auto">
      {/* ✅ Fixe le header en haut */}
      <stackLayout row="0" className="gradient-header p-6">
        <gridLayout rows="auto, auto" columns="*, auto" className="mb-4">
          <label
            text={`Zone ${zoneData.name}`}
            className="text-2xl font-bold text-white"
          />
        </gridLayout>
      </stackLayout>

      {/* ✅ Ajoute un scroll uniquement pour le contenu */}
      <scrollView row="1" className="page">
        <stackLayout>
          {/* ✅ Correction de la carte : Ajustement de la hauteur et suppression de la marge négative */}
          <gridLayout className="mx-4">
            <stackLayout className="card p-0">
              <webView src={mapUrl} className="rounded-lg h-64 w-full" />
            </stackLayout>
          </gridLayout>

          {/* ✅ Statistiques dynamiques */}
          <gridLayout columns="*, *" rows="auto" className="m-4">
            <stackLayout col="0" className="stat-card m-2 p-4">
              <label
                text={zoneData.available.toString()}
                className="text-2xl font-bold text-blue-600"
              />
              <label text="Places libres" className="text-sm text-gray-600" />
            </stackLayout>
            <stackLayout col="1" className="stat-card m-2 p-4">
              <label
                text={zoneData.averageTime}
                className="text-2xl font-bold text-green-600"
              />
              <label text="Temps moyen" className="text-sm text-gray-600" />
            </stackLayout>
          </gridLayout>

          {/* ✅ Informations dynamiques */}
          <stackLayout className="m-4">
            <label
              text="Informations"
              className="text-xl font-bold text-gray-800 mb-4"
            />
            <stackLayout className="card p-4">
              <gridLayout
                rows="auto, auto, auto"
                columns="24, *"
                className="gap-4"
              >
                <label row="0" col="0" text="♿" className="text-lg" />
                <label
                  row="0"
                  col="1"
                  text={
                    zoneData.covered ? "Place Handicapé" : "Parking non couvert"
                  }
                  className="text-gray-700"
                />

                <label row="1" col="0" text="⚡" className="text-lg" />
                <label
                  row="1"
                  col="1"
                  text={`${zoneData.chargingStations} bornes de recharge`}
                  className="text-gray-700"
                />

                <label row="2" col="0" text="🎫" className="text-lg" />
                <label
                  row="2"
                  col="1"
                  text={`Tarif: ${zoneData.pricePerHour}€/heure`}
                  className="text-gray-700"
                />
                
                

              </gridLayout>
            </stackLayout>
          </stackLayout>
        </stackLayout>
      </scrollView>
    </gridLayout>
  );
}
