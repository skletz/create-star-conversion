/// <reference path="../resources/d_ts/phaser.d.ts"/>
module Config
{
    // paths
    export const ASSETS_PATH: string = "assets/";
    export const GFX_PATH: string = ASSETS_PATH+"gfx/";
    export const AUDIO_PATH: string = ASSETS_PATH+"audio/";
    export const SFX_PATH: string = AUDIO_PATH+"sfx/";
    export const SPRITE_SHEETS_PATH: string = GFX_PATH+"spritesheets/";
    export const FONTS_PATH: string = ASSETS_PATH+"fonts/";

    // options
    export const DEBUG: boolean = true;
    export const TEST_LOADING = false; // test loading animation
    export const MAX_WIDTH: number = 1920;
    export const MAX_HEIGHT: number = 1080;
}



