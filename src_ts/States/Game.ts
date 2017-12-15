/// <reference path="../Entities/Character.ts"/>
module State
{
    export class Game extends Phaser.State
    {
        game: Phaser.Game;

        // sprites
        bg: Phaser.TileSprite;
        character: Sprite.Character;

        // groups
        bgGroup: Phaser.Group;
        frontGroup: Phaser.Group;

        music: Phaser.Sound;
        
        stateChangeSignal: Phaser.Signal;

        create()
        {

            // groups for sprite layering (prevents stacking by add order)
            this.bgGroup = this.game.add.group();
            this.frontGroup = this.game.add.group();

            // background
            this.bg = this.game.add.tileSprite(0, 0, this.game.cache.getImage('bg_game').width, this.game.cache.getImage('bg_game').height, 'bg_game');
            this.bg.scale.x = Utils.getProportionalScale(this.game.width, this.game.cache.getImage('bg_game').width);
            this.bg.scale.y = Utils.getProportionalScale(this.game.height, this.game.cache.getImage('bg_game').height);
            this.bgGroup.add(this.bg);

            // character
            this.character = new Sprite.Character(this.game, this.game.width - this.game.width/2, this.game.height - this.game.height/2);
            this.frontGroup.add(this.character);

            // start arcade physics
            //this.game.physics.startSystem(Phaser.Physics.ARCADE);
            //this.game.physics.arcade.gravity.y = 200;

            // sound
            this.music  = this.game.add.audio('music', 1,true); // looping music
            this.music.play();

            // stop all sounds on switching to different state    
            this.game.state.onStateChange.addOnce(this.stopSounds);
        }
        
        public stopSounds = () =>
        {            
            console.log("Sounds stopped!");
            this.sound.stopAll();
        }


        update()
        {

        }


        /* DEBUG (Show Bounding Box) */
        render()
        {
             this.game.debug.bodyInfo(this.character, 32, 32);    
             var charact = <Phaser.Sprite>this.frontGroup.getAt(0);
             if (charact != null) this.game.debug.body(charact);
        }
        
    }
}