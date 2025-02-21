import { Style } from "./map-view";
import { View, Image, LayoutBase, Property, Color, Builder, eachDescendant, ProxyViewContainer, StackLayout } from "@nativescript/core";
function onInfoWindowTemplatesChanged(mapView) {
    let _infoWindowTemplates = new Array();
    if (mapView.infoWindowTemplates && typeof mapView.infoWindowTemplates === "string") {
        _infoWindowTemplates = _infoWindowTemplates.concat(Builder.parseMultipleTemplates(mapView.infoWindowTemplates));
    }
    else if (mapView.infoWindowTemplates) {
        _infoWindowTemplates = _infoWindowTemplates.concat(mapView.infoWindowTemplates);
    }
    mapView._infoWindowTemplates = _infoWindowTemplates;
}
function onMapPropertyChanged(mapView) {
    if (!mapView.processingCameraEvent)
        mapView.updateCamera();
}
function onSetMinZoomMaxZoom(mapView) {
    mapView.setMinZoomMaxZoom();
}
function onPaddingPropertyChanged(mapView) {
    mapView.updatePadding();
}
function paddingValueConverter(value) {
    if (!Array.isArray(value)) {
        value = String(value).split(',');
    }
    value = value.map((v) => parseInt(v, 10));
    if (value.length >= 4) {
        return value;
    }
    else if (value.length === 3) {
        return [value[0], value[1], value[2], value[2]];
    }
    else if (value.length === 2) {
        return [value[0], value[0], value[1], value[1]];
    }
    else if (value.length === 1) {
        return [value[0], value[0], value[0], value[0]];
    }
    else {
        return [0, 0, 0, 0];
    }
}
function onDescendantsLoaded(view, callback) {
    if (!view)
        return callback();
    let loadingCount = 1;
    let loadedCount = 0;
    const watchLoaded = (view, event) => {
        const onLoaded = () => {
            view.off(event, onLoaded);
            loadedCount++;
            if (view instanceof Image && view.isLoading) {
                loadingCount++;
                watchLoaded(view, 'isLoadingChange');
                if (view.nativeView.onAttachedToWindow) {
                    view.nativeView.onAttachedToWindow();
                }
            }
            if (loadedCount === loadingCount)
                callback();
        };
        view.on(event, onLoaded);
    };
    eachDescendant(view, (descendant) => {
        loadingCount++;
        watchLoaded(descendant, View.loadedEvent);
        return true;
    });
    watchLoaded(view, View.loadedEvent);
}
export { Style as StyleBase };
export var knownTemplates;
(function (knownTemplates) {
    knownTemplates.infoWindowTemplate = "infoWindowTemplate";
})(knownTemplates || (knownTemplates = {}));
export var knownMultiTemplates;
(function (knownMultiTemplates) {
    knownMultiTemplates.infoWindowTemplates = "infoWindowTemplates";
})(knownMultiTemplates || (knownMultiTemplates = {}));
export function getColorHue(color) {
    if (typeof color === 'number') {
        while (color < 0) {
            color += 360;
        }
        return color % 360;
    }
    if (typeof color === 'string')
        color = new Color(color);
    if (!(color instanceof Color))
        return color;
    let min, max, delta, hue;
    const r = Math.max(0, Math.min(1, color.r / 255));
    const g = Math.max(0, Math.min(1, color.g / 255));
    const b = Math.max(0, Math.min(1, color.b / 255));
    min = Math.min(r, g, b);
    max = Math.max(r, g, b);
    delta = max - min;
    if (delta == 0) {
        hue = 0;
    }
    else if (r == max) {
        hue = (g - b) / delta;
    }
    else if (g == max) {
        hue = 2 + (b - r) / delta;
    }
    else {
        hue = 4 + (r - g) / delta;
    }
    hue = ((hue * 60) + 360) % 360;
    return hue;
}
export class MapViewBase extends View {
    constructor() {
        super(...arguments);
        this._markers = new Array();
        this._shapes = new Array();
        this._defaultInfoWindowTemplate = {
            key: "",
            createView: () => {
                if (this.infoWindowTemplate) {
                    let v = Builder.parse(this.infoWindowTemplate, this);
                    return v;
                }
                return undefined;
            }
        };
        this._infoWindowTemplates = new Array();
    }
    get gMap() {
        return this._gMap;
    }
    get processingCameraEvent() {
        return this._processingCameraEvent;
    }
    _getMarkerInfoWindowContent(marker) {
        var view;
        if (marker && marker._infoWindowView) {
            view = marker._infoWindowView;
            return view;
        }
        const template = this._getInfoWindowTemplate(marker);
        if (template)
            view = template.createView();
        if (!view)
            return null;
        if (!(view instanceof LayoutBase) ||
            view instanceof ProxyViewContainer) {
            let sp = new StackLayout();
            sp.addChild(view);
            view = sp;
        }
        marker._infoWindowView = view;
        view.bindingContext = marker;
        onDescendantsLoaded(view, () => {
            marker.hideInfoWindow();
            marker.showInfoWindow();
        });
        this._addView(view);
        view.onLoaded();
        return view;
    }
    _unloadInfoWindowContent(marker) {
        if (marker._infoWindowView) {
            marker._infoWindowView.onUnloaded();
            marker._infoWindowView = null;
        }
    }
    _getInfoWindowTemplate(marker) {
        if (marker) {
            const templateKey = marker.infoWindowTemplate;
            for (let i = 0, length = this._infoWindowTemplates.length; i < length; i++) {
                if (this._infoWindowTemplates[i].key === templateKey) {
                    return this._infoWindowTemplates[i];
                }
            }
        }
        return this._defaultInfoWindowTemplate;
    }
    removeAllPolylines() {
        if (!this._shapes)
            return null;
        this._shapes.forEach(shape => {
            if (shape.shape === 'polyline') {
                this.removeShape(shape);
            }
        });
    }
    removeAllPolygons() {
        if (!this._shapes)
            return null;
        this._shapes.forEach(shape => {
            if (shape.shape === 'polygon') {
                this.removeShape(shape);
            }
        });
    }
    removeAllCircles() {
        if (!this._shapes)
            return null;
        this._shapes.forEach(shape => {
            if (shape.shape === 'circle') {
                this.removeShape(shape);
            }
        });
    }
    notifyMapReady() {
        this.notify({ eventName: MapViewBase.mapReadyEvent, object: this, gMap: this.gMap });
    }
    notifyMarkerEvent(eventName, marker) {
        let args = { eventName: eventName, object: this, marker: marker };
        this.notify(args);
    }
    notifyShapeEvent(eventName, shape) {
        let args = { eventName: eventName, object: this, shape: shape };
        this.notify(args);
    }
    notifyMarkerTapped(marker) {
        this.notifyMarkerEvent(MapViewBase.markerSelectEvent, marker);
    }
    notifyMarkerInfoWindowTapped(marker) {
        this.notifyMarkerEvent(MapViewBase.markerInfoWindowTappedEvent, marker);
    }
    notifyMarkerInfoWindowClosed(marker) {
        this.notifyMarkerEvent(MapViewBase.markerInfoWindowClosedEvent, marker);
    }
    notifyShapeTapped(shape) {
        this.notifyShapeEvent(MapViewBase.shapeSelectEvent, shape);
    }
    notifyMarkerBeginDragging(marker) {
        this.notifyMarkerEvent(MapViewBase.markerBeginDraggingEvent, marker);
    }
    notifyMarkerEndDragging(marker) {
        this.notifyMarkerEvent(MapViewBase.markerEndDraggingEvent, marker);
    }
    notifyMarkerDrag(marker) {
        this.notifyMarkerEvent(MapViewBase.markerDragEvent, marker);
    }
    notifyPositionEvent(eventName, position) {
        let args = { eventName: eventName, object: this, position: position };
        this.notify(args);
    }
    notifyCameraEvent(eventName, camera) {
        let args = { eventName: eventName, object: this, camera: camera };
        this.notify(args);
    }
    notifyMyLocationTapped() {
        this.notify({ eventName: MapViewBase.myLocationTappedEvent, object: this });
    }
    notifyBuildingFocusedEvent(indoorBuilding) {
        let args = { eventName: MapViewBase.indoorBuildingFocusedEvent, object: this, indoorBuilding: indoorBuilding };
        this.notify(args);
    }
    notifyIndoorLevelActivatedEvent(activateLevel) {
        let args = { eventName: MapViewBase.indoorLevelActivatedEvent, object: this, activateLevel: activateLevel };
        this.notify(args);
    }
}
MapViewBase.mapReadyEvent = "mapReady";
MapViewBase.markerSelectEvent = "markerSelect";
MapViewBase.markerInfoWindowTappedEvent = "markerInfoWindowTapped";
MapViewBase.markerInfoWindowClosedEvent = "markerInfoWindowClosed";
MapViewBase.shapeSelectEvent = "shapeSelect";
MapViewBase.markerBeginDraggingEvent = "markerBeginDragging";
MapViewBase.markerEndDraggingEvent = "markerEndDragging";
MapViewBase.markerDragEvent = "markerDrag";
MapViewBase.coordinateTappedEvent = "coordinateTapped";
MapViewBase.coordinateLongPressEvent = "coordinateLongPress";
MapViewBase.cameraChangedEvent = "cameraChanged";
MapViewBase.cameraMoveEvent = "cameraMove";
MapViewBase.myLocationTappedEvent = "myLocationTapped";
MapViewBase.indoorBuildingFocusedEvent = "indoorBuildingFocused";
MapViewBase.indoorLevelActivatedEvent = "indoorLevelActivated";
export const infoWindowTemplateProperty = new Property({ name: "infoWindowTemplate" });
infoWindowTemplateProperty.register(MapViewBase);
export const infoWindowTemplatesProperty = new Property({ name: "infoWindowTemplates", valueChanged: onInfoWindowTemplatesChanged });
infoWindowTemplatesProperty.register(MapViewBase);
export const latitudeProperty = new Property({ name: 'latitude', defaultValue: 0, valueChanged: onMapPropertyChanged });
latitudeProperty.register(MapViewBase);
export const longitudeProperty = new Property({ name: 'longitude', defaultValue: 0, valueChanged: onMapPropertyChanged });
longitudeProperty.register(MapViewBase);
export const bearingProperty = new Property({ name: 'bearing', defaultValue: 0, valueChanged: onMapPropertyChanged });
bearingProperty.register(MapViewBase);
export const zoomProperty = new Property({ name: 'zoom', defaultValue: 0, valueChanged: onMapPropertyChanged });
zoomProperty.register(MapViewBase);
export const minZoomProperty = new Property({ name: 'minZoom', defaultValue: 0, valueChanged: onSetMinZoomMaxZoom });
minZoomProperty.register(MapViewBase);
export const maxZoomProperty = new Property({ name: 'maxZoom', defaultValue: 22, valueChanged: onSetMinZoomMaxZoom });
maxZoomProperty.register(MapViewBase);
export const tiltProperty = new Property({ name: 'tilt', defaultValue: 0, valueChanged: onMapPropertyChanged });
tiltProperty.register(MapViewBase);
export const paddingProperty = new Property({ name: 'padding', valueChanged: onPaddingPropertyChanged, valueConverter: paddingValueConverter });
paddingProperty.register(MapViewBase);
export const mapAnimationsEnabledProperty = new Property({ name: 'mapAnimationsEnabled', defaultValue: true });
mapAnimationsEnabledProperty.register(MapViewBase);
export class ProjectionBase {
}
export class VisibleRegionBase {
}
export class PositionBase {
}
export class BoundsBase {
}
export class MarkerBase {
}
export class ShapeBase {
}
export class PolylineBase extends ShapeBase {
    constructor() {
        super(...arguments);
        this.shape = 'polyline';
    }
    addPoint(point) {
        this._points.push(point);
        this.reloadPoints();
    }
    addPoints(points) {
        this._points = this._points.concat(points);
        this.reloadPoints();
    }
    removePoint(point) {
        var index = this._points.indexOf(point);
        if (index > -1) {
            this._points.splice(index, 1);
            this.reloadPoints();
        }
    }
    removeAllPoints() {
        this._points.length = 0;
        this.reloadPoints();
    }
    getPoints() {
        return this._points.slice();
    }
}
export class PolygonBase extends ShapeBase {
    constructor() {
        super(...arguments);
        this.shape = 'polygon';
    }
    addPoint(point) {
        this._points.push(point);
        this.reloadPoints();
    }
    addPoints(points) {
        this._points = this._points.concat(points);
        this.reloadPoints();
    }
    removePoint(point) {
        var index = this._points.indexOf(point);
        if (index > -1) {
            this._points.splice(index, 1);
            this.reloadPoints();
        }
    }
    removeAllPoints() {
        this._points.length = 0;
        this.reloadPoints();
    }
    getPoints() {
        return this._points.slice();
    }
    addHole(hole) {
        this._holes.push(hole);
        this.reloadHoles();
    }
    addHoles(holes) {
        this._holes = this._holes.concat(holes);
        this.reloadHoles();
    }
    removeHole(hole) {
        var index = this._holes.indexOf(hole);
        if (index > -1) {
            this._holes.splice(index, 1);
            this.reloadHoles();
        }
    }
    removeAllHoles() {
        this._holes.length = 0;
        this.reloadHoles();
    }
    getHoles() {
        return this._holes.slice();
    }
}
export class CircleBase extends ShapeBase {
    constructor() {
        super(...arguments);
        this.shape = 'circle';
    }
}
