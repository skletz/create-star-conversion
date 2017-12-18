/// <reference path="../../resources/d_ts/phaser.d.ts"/>
module State
{
    export class Menu extends Phaser.State
    {
        game: Phaser.Game;
        
        bg: Phaser.TileSprite;

        create()
        {

            this.bg = this.game.add.tileSprite(0, 0, this.game.cache.getImage('bg_menu').width, this.game.cache.getImage('bg_menu').height, 'bg_menu');
            this.bg.scale.x = Utils.getProportionalScale(this.game.width, this.game.cache.getImage('bg_menu').width);
            this.bg.scale.y = Utils.getProportionalScale(this.game.height, this.game.cache.getImage('bg_menu').height);
            
        }

        update()
        {
            
        }
    }
}