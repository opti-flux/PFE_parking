import * as React from 'react';
import * as ReactNativeScript from 'react-nativescript';
import { MainStack } from './components/MainStack';

Object.defineProperty(global, '__DEV__', { value: false });

ReactNativeScript.start(React.createElement(MainStack, {}, null));