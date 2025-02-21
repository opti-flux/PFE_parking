import * as React from "react";
import { FrameNavigationProp } from "react-nativescript-navigation";
import { MainStackParamList } from "../NavigationParamList";

type MapScreenProps = {
  navigation: FrameNavigationProp<MainStackParamList, "Carte">;
};

export function MapScreen({ navigation }: MapScreenProps) {
  const parkingData = require("../assets/parkingData.json"); //  Charger les données localement

  // Générer l'URL avec toutes les zones
  const zonesParams = encodeURIComponent(JSON.stringify(parkingData.zones));
  const mapUrl = `~/assets/map.html?zones=${zonesParams}`;

  return (
    <gridLayout rows="auto, *, auto">
      {/*  Header */}
      <stackLayout row="0" className="gradient-header p-6">
        <label
          text="Carte des parkings"
          className="text-white text-center text-2xl font-bold"
        />
      </stackLayout>

      {/*  Carte interactive */}
      <stackLayout row="1" className="m-4">
        <webView src={mapUrl} className="rounded-lg h-full w-full" />
      </stackLayout>

      {/*  Retour à l'accueil */}
      <stackLayout row="2" className="bg-white p-4">
        <button
          text="Retour"
          onTap={() => navigation.goBack()}
          className="text-lg text-blue-600"
        />
      </stackLayout>
    </gridLayout>
  );
}
