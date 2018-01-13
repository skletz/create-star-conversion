/// <reference path="../../resources/d_ts/phaser.d.ts"/>
module State
{
    export class Title extends Phaser.State
    {
        game: Phaser.Game;

        bg: Phaser.TileSprite;

        create()
        {
            this.bg = this.game.add.tileSprite(0, 0, this.game.cache.getImage('bg_title').width, this.game.cache.getImage('bg_title').height, 'bg_title');
            this.bg.scale.x = Utils.getProportionalScale(this.game.width, this.game.cache.getImage('bg_title').width);
            this.bg.scale.y = Utils.getProportionalScale(this.game.height, this.game.cache.getImage('bg_title').height);


            // debug keys
            // if (Config.DEBUG)
            // {
            //     console.log("moo");
            //     var titleButton = this.game.input.keyboard.addKey(Phaser.Keyboard.F1);
            //     titleButton.onDown.add(() => this.game.state.start("TitleState"));
            //     var menuButton = this.game.input.keyboard.addKey(Phaser.Keyboard.F2);
            //     menuButton.onDown.add(() => this.game.state.start("MenuState"));
            //     var gameButton = this.game.input.keyboard.addKey(Phaser.Keyboard.F3);
            //     gameButton.onDown.add(() => this.game.state.start("GameState"));
            //     var endButton = this.game.input.keyboard.addKey(Phaser.Keyboard.F4);
            //     endButton.onDown.add(() => this.game.state.start("EndState"));
            //
            //     this.game.input.resetLocked = true; // with this input does not get reset on state change
            // }
            
        }

        
        
    }    
}