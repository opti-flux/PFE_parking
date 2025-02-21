import { BaseNavigationContainer } from "@react-navigation/core";
import * as React from "react";
import { stackNavigatorFactory } from "react-nativescript-navigation";
import { HomeScreen } from "./HomeScreen";
import { ZoneDetailsScreen } from "./ZoneDetailsScreen";
import { MapScreen } from "./MapScreen"; // ✅ Importer MapScreen

const StackNavigator = stackNavigatorFactory();

export const MainStack = () => (
  <BaseNavigationContainer>
    <StackNavigator.Navigator
      initialRouteName="OptiFeux"
      screenOptions={{
        headerStyle: {
          backgroundColor: "#007AFF",
        },
        headerTintColor: "#ffffff",
        headerShown: true,
      }}
    >
      <StackNavigator.Screen name="OptiFeux" component={HomeScreen} />
      <StackNavigator.Screen
        name="Zone Details"
        component={ZoneDetailsScreen}
      />
      <StackNavigator.Screen
        name="Carte"
        component={MapScreen} // ✅ Ajout de la page "Carte"
      />
    </StackNavigator.Navigator>
  </BaseNavigationContainer>
);
