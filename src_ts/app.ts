/// <reference path="../resources/d_ts/phaser.d.ts"/>
module Main
{
    export class GameTemplate
    {
        constructor(width:number, height:number)
        {
            if (width > Config.MAX_WIDTH) width = Config.MAX_WIDTH;
            if (height > Config.MAX_HEIGHT) height = Config.MAX_HEIGHT;

            // create game in div with 'content' id
            this.game = new Phaser.Game(width, height, Phaser.AUTO, 'content', {create: this.create, preload: this.preload });
        }

        game: Phaser.Game;

        preload()
        {

            this.game.load.image('progressBar', 'assets/gfx/sprites/progressbar.png');
            this.game.load.image('progressBarOutline', 'assets/gfx/sprites/progressbar_outline.png');

        }



        create()
        {

            /** phaser settings **/
            this.game.stage.backgroundColor = '#525252';
            // use arcade physics engine
            this.game.physics.startSystem(Phaser.Physics.ARCADE);
            // ensure sprites are rendered at integer positions:
            // sprites rendered at non-integer (sub-pixel) positions appear blurry,
            // as canvas tries to anti-alias them between the two pixels.
            // IMPORTANT: STATES DONT render anything nor do they
            // have any Display properties, hence renderer has to be set via 'this.game'
            this.game.renderer.renderSession.roundPixels = true;

            // scale according to window
            this.game.scale.scaleMode = Phaser.ScaleManager.NO_SCALE;

            // game states
            this.game.state.add("LoadState", State.Load, true); // start with load
            this.game.state.add("TitleState", State.Title, false);
            this.game.state.add("MenuState", State.Menu, false);
            this.game.state.add("GameState", State.Game, false);
            this.game.state.add("EndState", State.End, false);
        }




        public getGame()
        {
            return this.game;
        }

    }
}


// Create game class = starting game
var gameTemplate: Main.GameTemplate;

window.onload = () =>
{
    var height = window.innerHeight;
    var width = window.innerWidth;
    gameTemplate = new Main.GameTemplate(width, height); // create game

};

var resizeId;

// reloads game when resizing (for adjusting dimensions)
window.onresize = function(event)
{
    // timeout for window resizing
    clearTimeout(resizeId);
    resizeId = setTimeout(finishedResize, 500);

};

function finishedResize()
{
    //console.log("window was resized!");
    location.reload(); // reload page to adjust to resolution
}
