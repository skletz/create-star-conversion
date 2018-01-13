/// <reference path="../resources/d_ts/phaser.d.ts"/>
var Config;
(function (Config) {
    // paths
    Config.ASSETS_PATH = "assets/";
    Config.GFX_PATH = Config.ASSETS_PATH + "gfx/";
    Config.AUDIO_PATH = Config.ASSETS_PATH + "audio/";
    Config.SFX_PATH = Config.AUDIO_PATH + "sfx/";
    Config.SPRITE_SHEETS_PATH = Config.GFX_PATH + "spritesheets/";
    Config.FONTS_PATH = Config.ASSETS_PATH + "fonts/";
    // options
    Config.DEBUG = true;
    Config.TEST_LOADING = false; // test loading animation
    Config.MAX_WIDTH = 1920;
    Config.MAX_HEIGHT = 1080;
})(Config || (Config = {}));
//# sourceMappingURL=config.js.map