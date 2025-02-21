import { MapViewBase, BoundsBase, CircleBase, MarkerBase, PolygonBase, PolylineBase, ProjectionBase, PositionBase, latitudeProperty, VisibleRegionBase, longitudeProperty, bearingProperty, zoomProperty, tiltProperty, getColorHue } from "./map-view-common";
import { GC, layout } from "@nativescript/core/utils";
import { Image, ImageSource } from "@nativescript/core";
export * from "./map-view-common";
let IndoorDisplayDelegateImpl = IndoorDisplayDelegateImpl_1 = class IndoorDisplayDelegateImpl extends NSObject {
    static initWithOwner(owner) {
        let handler = IndoorDisplayDelegateImpl_1.new();
        handler._owner = owner;
        return handler;
    }
    didChangeActiveBuilding(indoorBuilding) {
        let owner = this._owner.get();
        if (owner) {
            let data = null;
            if (indoorBuilding) {
                const levels = [];
                let count = 0;
                while (count < indoorBuilding.levels.count) {
                    levels.push({
                        name: indoorBuilding.levels[count].name,
                        shortName: indoorBuilding.levels[count].shortName,
                    });
                    count += 1;
                }
                data = {
                    defaultLevelIndex: indoorBuilding.defaultLevelIndex,
                    levels: levels,
                    isUnderground: indoorBuilding.underground,
                };
            }
            owner.notifyBuildingFocusedEvent(data);
        }
    }
    didChangeActiveLevel(activateLevel) {
        let owner = this._owner.get();
        if (owner) {
            let data = null;
            if (activateLevel) {
                data = {
                    name: activateLevel.name,
                    shortName: activateLevel.shortName,
                };
            }
            owner.notifyIndoorLevelActivatedEvent(data);
        }
    }
};
IndoorDisplayDelegateImpl.ObjCProtocols = [GMSIndoorDisplayDelegate];
IndoorDisplayDelegateImpl = IndoorDisplayDelegateImpl_1 = __decorate([
    NativeClass()
], IndoorDisplayDelegateImpl);
let MapViewDelegateImpl = MapViewDelegateImpl_1 = class MapViewDelegateImpl extends NSObject {
    static initWithOwner(owner) {
        let handler = MapViewDelegateImpl_1.new();
        handler._owner = owner;
        return handler;
    }
    mapViewIdleAtCameraPosition(mapView, cameraPosition) {
        let owner = this._owner.get();
        if (owner) {
            owner._processingCameraEvent = true;
            let cameraChanged = false;
            if (owner.latitude != cameraPosition.target.latitude) {
                cameraChanged = true;
                latitudeProperty.nativeValueChange(owner, cameraPosition.target.latitude);
            }
            if (owner.longitude != cameraPosition.target.longitude) {
                cameraChanged = true;
                longitudeProperty.nativeValueChange(owner, cameraPosition.target.longitude);
            }
            if (owner.bearing != cameraPosition.bearing) {
                cameraChanged = true;
                bearingProperty.nativeValueChange(owner, cameraPosition.bearing);
            }
            if (owner.zoom != cameraPosition.zoom) {
                cameraChanged = true;
                zoomProperty.nativeValueChange(owner, cameraPosition.zoom);
            }
            if (owner.tilt != cameraPosition.viewingAngle) {
                cameraChanged = true;
                tiltProperty.nativeValueChange(owner, cameraPosition.viewingAngle);
            }
            if (cameraChanged) {
                owner.notifyCameraEvent(MapViewBase.cameraChangedEvent, {
                    latitude: cameraPosition.target.latitude,
                    longitude: cameraPosition.target.longitude,
                    zoom: cameraPosition.zoom,
                    bearing: cameraPosition.bearing,
                    tilt: cameraPosition.viewingAngle
                });
            }
            owner._processingCameraEvent = false;
        }
    }
    mapViewDidChangeCameraPosition(mapView, cameraPosition) {
        let owner = this._owner.get();
        owner.notifyCameraEvent(MapViewBase.cameraMoveEvent, {
            latitude: cameraPosition.target.latitude,
            longitude: cameraPosition.target.longitude,
            zoom: cameraPosition.zoom,
            bearing: cameraPosition.bearing,
            tilt: cameraPosition.viewingAngle
        });
    }
    mapViewDidTapAtCoordinate(mapView, coordinate) {
        let owner = this._owner.get();
        if (owner) {
            let position = Position.positionFromLatLng(coordinate.latitude, coordinate.longitude);
            owner.notifyPositionEvent(MapViewBase.coordinateTappedEvent, position);
        }
    }
    mapViewDidLongPressAtCoordinate(mapView, coordinate) {
        let owner = this._owner.get();
        if (owner) {
            let position = Position.positionFromLatLng(coordinate.latitude, coordinate.longitude);
            owner.notifyPositionEvent(MapViewBase.coordinateLongPressEvent, position);
        }
    }
    mapViewDidTapMarker(mapView, gmsMarker) {
        const owner = this._owner.get();
        if (owner) {
            let marker = owner.findMarker((marker) => marker.ios == gmsMarker);
            if (marker) {
                owner.notifyMarkerTapped(marker);
            }
        }
        return false;
    }
    mapViewDidTapOverlay(mapView, gmsOverlay) {
        let owner = this._owner.get();
        if (owner) {
            let shape = owner.findShape((shape) => shape.ios == gmsOverlay);
            if (shape) {
                owner.notifyShapeTapped(shape);
            }
        }
    }
    mapViewDidBeginDraggingMarker(mapView, gmsMarker) {
        let owner = this._owner.get();
        if (owner) {
            let marker = owner.findMarker((marker) => marker.ios == gmsMarker);
            owner.notifyMarkerBeginDragging(marker);
        }
    }
    mapViewDidEndDraggingMarker(mapView, gmsMarker) {
        let owner = this._owner.get();
        if (owner) {
            let marker = owner.findMarker((marker) => marker.ios == gmsMarker);
            owner.notifyMarkerEndDragging(marker);
        }
    }
    mapViewDidDragMarker(mapView, gmsMarker) {
        let owner = this._owner.get();
        if (owner) {
            let marker = owner.findMarker((marker) => marker.ios == gmsMarker);
            owner.notifyMarkerDrag(marker);
        }
    }
    mapViewDidTapInfoWindowOfMarker(mapView, gmsMarker) {
        let owner = this._owner.get();
        if (owner) {
            let marker = owner.findMarker((marker) => marker.ios == gmsMarker);
            owner.notifyMarkerInfoWindowTapped(marker);
        }
    }
    mapViewDidCloseInfoWindowOfMarker(mapView, gmsMarker) {
        let owner = this._owner.get();
        if (owner) {
            let marker = owner.findMarker((marker) => marker.ios == gmsMarker);
            owner.notifyMarkerInfoWindowClosed(marker);
        }
    }
    didTapMyLocationButtonForMapView(mapView) {
        const owner = this._owner.get();
        if (owner) {
            owner.notifyMyLocationTapped();
            return false;
        }
        return true;
    }
    mapViewMarkerInfoWindow(mapView, gmsMarker) {
        return null;
    }
    mapViewMarkerInfoContents(mapView, gmsMarker) {
        let owner = this._owner.get();
        if (!owner)
            return null;
        let marker = owner.findMarker((marker) => marker.ios == gmsMarker);
        var content = owner._getMarkerInfoWindowContent(marker);
        if (content) {
            let width = Number(content.width);
            if (Number.isNaN(width))
                width = null;
            let height = Number(content.height);
            if (Number.isNaN(height))
                height = null;
            if (!height || !width) {
                const bounds = UIScreen.mainScreen.bounds;
                width = width || (bounds.size.width * .7);
                height = height || (bounds.size.height * .4);
            }
            this._layoutRootView(content, CGRectMake(0, 0, width, height));
            return content.ios;
        }
        return null;
    }
    _layoutRootView(rootView, parentBounds) {
        if (!rootView || !parentBounds) {
            return;
        }
        const size = parentBounds.size;
        const width = layout.toDevicePixels(size.width);
        const height = layout.toDevicePixels(size.height);
        const widthSpec = layout.makeMeasureSpec(width, layout.EXACTLY);
        const heightSpec = layout.makeMeasureSpec(height, layout.EXACTLY);
        rootView.measure(widthSpec, heightSpec);
        const origin = parentBounds.origin;
        const left = origin.x;
        const top = origin.y;
        rootView.layout(left, top, width, height);
    }
};
MapViewDelegateImpl.ObjCProtocols = [GMSMapViewDelegate];
MapViewDelegateImpl = MapViewDelegateImpl_1 = __decorate([
    NativeClass()
], MapViewDelegateImpl);
export class MapView extends MapViewBase {
    constructor() {
        super();
        this._markers = new Array();
        this.nativeView = GMSMapView.mapWithFrameCamera(CGRectZero, this._createCameraPosition());
        this._delegate = MapViewDelegateImpl.initWithOwner(new WeakRef(this));
        this._indoorDelegate = IndoorDisplayDelegateImpl.initWithOwner(new WeakRef(this));
        this.updatePadding();
    }
    onLoaded() {
        super.onLoaded();
        this.nativeView.delegate = this._delegate;
        this.nativeView.indoorDisplay.delegate = this._indoorDelegate;
        this.notifyMapReady();
    }
    onUnloaded() {
        this.nativeView.delegate = null;
        this.nativeView.indoorDisplay.delegate = null;
        super.onUnloaded();
    }
    disposeNativeView() {
        this._markers = null;
        this._delegate = null;
        this._indoorDelegate = null;
        super.disposeNativeView();
        GC();
    }
    ;
    _createCameraPosition() {
        return GMSCameraPosition.cameraWithLatitudeLongitudeZoomBearingViewingAngle(this.latitude, this.longitude, this.zoom, this.bearing, this.tilt);
    }
    updateCamera() {
        if (this.mapAnimationsEnabled) {
            this.nativeView.animateToCameraPosition(this._createCameraPosition());
        }
        else {
            this.nativeView.camera = this._createCameraPosition();
        }
    }
    setViewport(bounds, padding) {
        var p = UIEdgeInsetsMake(padding, padding, padding, padding) || this.gMap.padding;
        let cameraPosition = this.nativeView.cameraForBoundsInsets(bounds.ios, p);
        if (this.mapAnimationsEnabled) {
            this.nativeView.animateToCameraPosition(cameraPosition);
        }
        else {
            this.nativeView.camera = cameraPosition;
        }
    }
    updatePadding() {
        if (this.padding) {
            this.gMap.padding = UIEdgeInsetsMake(this.padding[0] || 0, this.padding[2] || 0, this.padding[1] || 0, this.padding[3] || 0);
        }
    }
    get ios() {
        throw new Error('Now use instance.nativeView instead of instance.ios');
    }
    get gMap() {
        return this.nativeView;
    }
    get projection() {
        return new Projection(this.nativeView.projection);
    }
    get settings() {
        return (this.nativeView) ? new UISettings(this.nativeView.settings) : null;
    }
    get myLocationEnabled() {
        return (this.nativeView) ? this.nativeView.myLocationEnabled : false;
    }
    set myLocationEnabled(value) {
        if (this.nativeView)
            this.nativeView.myLocationEnabled = value;
    }
    setMinZoomMaxZoom() {
        this.gMap.setMinZoomMaxZoom(this.minZoom, this.maxZoom);
    }
    addMarker(...markers) {
        if (!markers || !this._markers || !this.gMap)
            return null;
        markers.forEach(marker => {
            marker.ios.map = this.gMap;
            this._markers.push(marker);
        });
    }
    removeMarker(...markers) {
        if (!markers || !this._markers || !this.gMap)
            return null;
        markers.forEach(marker => {
            this._unloadInfoWindowContent(marker);
            marker.ios.map = null;
            this._markers.splice(this._markers.indexOf(marker), 1);
        });
    }
    removeAllMarkers() {
        if (!this._markers)
            return null;
        this._markers.forEach(marker => {
            this._unloadInfoWindowContent(marker);
            marker.ios.map = null;
        });
        this._markers = [];
    }
    findMarker(callback) {
        if (!this._markers)
            return null;
        return this._markers.find(callback);
    }
    addPolyline(shape) {
        if (!this._shapes)
            return null;
        shape.loadPoints();
        shape.ios.map = this.gMap;
        this._shapes.push(shape);
    }
    addPolygon(shape) {
        if (!this._shapes)
            return null;
        shape.ios.map = this.gMap;
        this._shapes.push(shape);
    }
    addCircle(shape) {
        if (!this._shapes)
            return null;
        shape.ios.map = this.gMap;
        this._shapes.push(shape);
    }
    removeShape(shape) {
        if (!this._shapes)
            return null;
        shape.ios.map = null;
        this._shapes.splice(this._shapes.indexOf(shape), 1);
    }
    removeAllShapes() {
        if (!this._shapes)
            return null;
        this._shapes.forEach(shape => {
            shape.ios.map = null;
        });
        this._shapes = [];
    }
    findShape(callback) {
        if (!this._shapes)
            return null;
        return this._shapes.find(callback);
    }
    clear() {
        this._markers = [];
        this.nativeView.clear();
    }
    setStyle(style) {
        try {
            this.nativeView.mapStyle = GMSMapStyle.styleWithJSONStringError(JSON.stringify(style));
            return true;
        }
        catch (err) {
            return false;
        }
    }
}
export class UISettings {
    get ios() {
        return this._ios;
    }
    constructor(ios) {
        this._ios = ios;
    }
    get compassEnabled() {
        return this._ios.compassButton;
    }
    set compassEnabled(value) {
        this._ios.compassButton = value;
    }
    get indoorLevelPickerEnabled() {
        return this._ios.indoorPicker;
    }
    set indoorLevelPickerEnabled(value) {
        this._ios.indoorPicker = value;
    }
    get mapToolbarEnabled() {
        return false;
    }
    set mapToolbarEnabled(value) {
        if (value)
            console.warn("Map toolbar not available on iOS");
    }
    get myLocationButtonEnabled() {
        return this._ios.myLocationButton;
    }
    set myLocationButtonEnabled(value) {
        this._ios.myLocationButton = value;
    }
    get rotateGesturesEnabled() {
        return this._ios.rotateGestures;
    }
    set rotateGesturesEnabled(value) {
        this._ios.rotateGestures = value;
    }
    get scrollGesturesEnabled() {
        return this._ios.scrollGestures;
    }
    set scrollGesturesEnabled(value) {
        this._ios.scrollGestures = value;
    }
    get tiltGesturesEnabled() {
        return this._ios.tiltGestures;
    }
    set tiltGesturesEnabled(value) {
        this._ios.tiltGestures = value;
    }
    get zoomControlsEnabled() {
        return false;
    }
    set zoomControlsEnabled(value) {
        if (value)
            console.warn("Zoom controls not available on iOS");
    }
    get zoomGesturesEnabled() {
        return this._ios.zoomGestures;
    }
    set zoomGesturesEnabled(value) {
        this._ios.zoomGestures = value;
    }
}
export class Projection extends ProjectionBase {
    get ios() {
        return this._ios;
    }
    get visibleRegion() {
        return new VisibleRegion(this.ios.visibleRegion());
    }
    fromScreenLocation(point) {
        var location = this.ios.coordinateForPoint(CGPointMake(point.x, point.y));
        return new Position(location);
    }
    toScreenLocation(position) {
        var cgPoint = this.ios.pointForCoordinate(position.ios);
        return {
            x: cgPoint.x,
            y: cgPoint.y
        };
    }
    constructor(ios) {
        super();
        this._ios = ios;
    }
}
export class VisibleRegion extends VisibleRegionBase {
    get ios() {
        return this._ios;
    }
    get nearLeft() {
        return new Position(this.ios.nearLeft);
    }
    get nearRight() {
        return new Position(this.ios.nearRight);
    }
    get farLeft() {
        return new Position(this.ios.farLeft);
    }
    get farRight() {
        return new Position(this.ios.farRight);
    }
    get bounds() {
        return new Bounds(GMSCoordinateBounds.alloc().initWithRegion(this.ios));
    }
    constructor(ios) {
        super();
        this._ios = ios;
    }
}
export class Bounds extends BoundsBase {
    get ios() {
        return this._ios;
    }
    get southwest() {
        return new Position(this.ios.southWest);
    }
    get northeast() {
        return new Position(this._ios.northEast);
    }
    constructor(ios) {
        super();
        this._ios = ios;
    }
    static fromCoordinates(southwest, northeast) {
        return new Bounds(GMSCoordinateBounds.alloc().initWithCoordinateCoordinate(southwest.ios, northeast.ios));
    }
}
export class Position extends PositionBase {
    get ios() {
        return this._ios;
    }
    get latitude() {
        return this._ios.latitude;
    }
    set latitude(latitude) {
        this._ios = CLLocationCoordinate2DMake(latitude, this.longitude);
    }
    get longitude() {
        return this._ios.longitude;
    }
    set longitude(longitude) {
        this._ios = CLLocationCoordinate2DMake(this.latitude, longitude);
    }
    constructor(ios) {
        super();
        this._ios = ios || CLLocationCoordinate2DMake(0, 0);
    }
    static positionFromLatLng(latitude, longitude) {
        let position = new Position();
        position.latitude = latitude;
        position.longitude = longitude;
        return position;
    }
}
export class Marker extends MarkerBase {
    constructor() {
        super();
        this._alpha = 1;
        this._visible = true;
        this._ios = GMSMarker.new();
    }
    static getIconForColor(hue) {
        const hueKey = hue.toFixed(8);
        if (!Marker.cachedColorIcons[hueKey]) {
            const icon = GMSMarker.markerImageWithColor(UIColor.colorWithHueSaturationBrightnessAlpha(hue, 1, 1, 1));
            Marker.cachedColorIcons[hueKey] = icon;
        }
        return Marker.cachedColorIcons[hueKey];
    }
    get position() {
        return new Position(this._ios.position);
    }
    set position(position) {
        this._ios.position = position.ios;
    }
    get rotation() {
        return this._ios.rotation;
    }
    set rotation(value) {
        this._ios.rotation = value;
    }
    get zIndex() {
        return this._ios.zIndex;
    }
    set zIndex(value) {
        this._ios.zIndex = value;
    }
    get title() {
        return this._ios.title;
    }
    set title(title) {
        this._ios.title = title;
    }
    get snippet() {
        return this._ios.snippet;
    }
    set snippet(snippet) {
        this._ios.snippet = snippet;
    }
    showInfoWindow() {
        this._ios.map.selectedMarker = this._ios;
    }
    isInfoWindowShown() {
        return this._ios.map.selectedMarker == this._ios;
    }
    hideInfoWindow() {
        this._ios.map.selectedMarker = null;
    }
    get color() {
        return this._color;
    }
    set color(value) {
        value = getColorHue(value);
        this._color = value;
        if (this._color) {
            this._ios.icon = Marker.getIconForColor(this._color / 360);
        }
        else {
            this._ios.icon = null;
        }
    }
    get icon() {
        return this._icon;
    }
    set icon(value) {
        if (typeof value === 'string') {
            var tempIcon = new Image();
            tempIcon.imageSource = ImageSource.fromResourceSync(String(value));
            value = tempIcon;
        }
        this._icon = value;
        this._ios.icon = (value) ? this._icon.imageSource.ios : null;
    }
    get alpha() {
        return this._alpha;
    }
    set alpha(value) {
        this._alpha = value;
        if (this._visible)
            this._ios.opacity = value;
    }
    get visible() {
        return this._visible;
    }
    set visible(value) {
        this._visible = value;
        this._ios.opacity = (this._visible) ? this._alpha : 0;
    }
    get flat() {
        return this._ios.flat;
    }
    set flat(value) {
        this._ios.flat = value;
    }
    get anchor() {
        return [this._ios.groundAnchor.x, this._ios.groundAnchor.y];
    }
    set anchor(value) {
        this._ios.groundAnchor = CGPointMake(value[0], value[1]);
    }
    get draggable() {
        return this._ios.draggable;
    }
    set draggable(value) {
        this._ios.draggable = value;
    }
    get ios() {
        return this._ios;
    }
}
Marker.cachedColorIcons = {};
export class Polyline extends PolylineBase {
    constructor() {
        super();
        this._ios = GMSPolyline.new();
        this._points = [];
    }
    get clickable() {
        return this._ios.tappable;
    }
    set clickable(value) {
        this._ios.tappable = value;
    }
    get zIndex() {
        return this._ios.zIndex;
    }
    set zIndex(value) {
        this._ios.zIndex = value;
    }
    loadPoints() {
        var points = GMSMutablePath.new();
        this._points.forEach(function (point) {
            points.addCoordinate(point.ios);
        }.bind(this));
        this._ios.path = points;
    }
    reloadPoints() {
        this.loadPoints();
    }
    get width() {
        return this._ios.strokeWidth;
    }
    set width(value) {
        this._ios.strokeWidth = value;
    }
    get color() {
        return this._color;
    }
    set color(value) {
        this._color = value;
        this._ios.strokeColor = value.ios;
    }
    get geodesic() {
        return this._ios.geodesic;
    }
    set geodesic(value) {
        this._ios.geodesic = value;
    }
    get ios() {
        return this._ios;
    }
}
export class Polygon extends PolygonBase {
    constructor() {
        super();
        this._ios = GMSPolygon.new();
        this._points = [];
        this._holes = [];
    }
    get clickable() {
        return this._ios.tappable;
    }
    set clickable(value) {
        this._ios.tappable = value;
    }
    get zIndex() {
        return this._ios.zIndex;
    }
    set zIndex(value) {
        this._ios.zIndex = value;
    }
    loadPoints() {
        var points = GMSMutablePath.new();
        this._points.forEach((point) => {
            points.addCoordinate(point.ios);
        });
        this._ios.path = points;
    }
    loadHoles() {
        var holes = [];
        this._holes.forEach((hole) => {
            var points = GMSMutablePath.new();
            hole.forEach((point) => {
                points.addCoordinate(point.ios);
            });
            holes.push(points);
        });
        this._ios.holes = holes;
    }
    reloadPoints() {
        this.loadPoints();
    }
    reloadHoles() {
        this.loadHoles();
    }
    get strokeWidth() {
        return this._ios.strokeWidth;
    }
    set strokeWidth(value) {
        this._ios.strokeWidth = value;
    }
    get strokeColor() {
        return this._strokeColor;
    }
    set strokeColor(value) {
        this._strokeColor = value;
        this._ios.strokeColor = value.ios;
    }
    get fillColor() {
        return this._fillColor;
    }
    set fillColor(value) {
        this._fillColor = value;
        this._ios.fillColor = value.ios;
    }
    get ios() {
        return this._ios;
    }
}
export class Circle extends CircleBase {
    constructor() {
        super();
        this._ios = GMSCircle.new();
    }
    get clickable() {
        return this._ios.tappable;
    }
    set clickable(value) {
        this._ios.tappable = value;
    }
    get zIndex() {
        return this._ios.zIndex;
    }
    set zIndex(value) {
        this._ios.zIndex = value;
    }
    get center() {
        return this._center;
    }
    set center(value) {
        this._center = value;
        this._ios.position = value.ios;
    }
    get radius() {
        return this._ios.radius;
    }
    set radius(value) {
        this._ios.radius = value;
    }
    get strokeWidth() {
        return this._ios.strokeWidth;
    }
    set strokeWidth(value) {
        this._ios.strokeWidth = value;
    }
    get strokeColor() {
        return this._strokeColor;
    }
    set strokeColor(value) {
        this._strokeColor = value;
        this._ios.strokeColor = value.ios;
    }
    get fillColor() {
        return this._fillColor;
    }
    set fillColor(value) {
        this._fillColor = value;
        this._ios.fillColor = value.ios;
    }
    get ios() {
        return this._ios;
    }
}
var IndoorDisplayDelegateImpl_1, MapViewDelegateImpl_1;
