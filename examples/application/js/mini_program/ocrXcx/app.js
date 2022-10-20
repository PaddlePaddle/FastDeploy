/* global wx, App */
import * as paddlejs from '@paddlejs/paddlejs-core';
import '@paddlejs/paddlejs-backend-webgl';
// eslint-disable-next-line no-undef
const plugin = requirePlugin('paddlejs-plugin');
plugin.register(paddlejs, wx);

App({
    globalData: {
        Paddlejs: paddlejs.Runner
    }
});
