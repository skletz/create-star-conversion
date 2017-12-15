/// <reference path="../../resources/d_ts/phaser.d.ts"/>
module State
{
    export class End extends Phaser.State
    {
        game: Phaser.Game;
        
        bg: Phaser.TileSprite;

        create()
        {
            this.bg = this.game.add.tileSprite(0, 0, this.game.cache.getImage('bg_end').width, this.game.cache.getImage('bg_end').height, 'bg_end');
            this.bg.scale.x = Utils.getProportionalScale(this.game.width, this.game.cache.getImage('bg_end').width);
            this.bg.scale.y = Utils.getProportionalScale(this.game.height, this.game.cache.getImage('bg_end').height);
        }

        update()
        {
            
        }
    }
}