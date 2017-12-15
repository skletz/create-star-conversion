/// <reference path="../../resources/d_ts/phaser.d.ts"/>
module Sprite
{
    export class Character extends Phaser.Sprite
    {

        public static MAX_SPEED: number = 30; // 30 fps

        game: Phaser.Game;
        animSpeed: number;

        cursors: Phaser.CursorKeys;

        effect: Phaser.Sound;

        constructor(game:Phaser.Game,x:number,y:number)
        {
            super(game,x,y,"BIRD_FLYING", 0); // frame 0

            this.anchor.setTo(.5, 1); //so it flips around its middle

            this.game = game;
            this.animSpeed = 10; // 10 fps

            // arcade physics
            this.game.physics.enable(this, Phaser.Physics.ARCADE);
            this.body.collideWorldBounds = true;
            this.body.bounce.set(0.4);
            this.body.setSize(74,66, 12, 10);
            //this.body.velocity.setTo(0, -20);

            // sfx
            this.effect = this.game.add.audio('ding',0.5,false);

            this.Animate();

            this.cursors = this.game.input.keyboard.createCursorKeys();

            // usual way of adding to game (adding to group makes this obsolete)
            //this.game.add.existing(this);
        }




        update()
        {
            this.handleMovement();


            this.handleCollisions();
        }

        private handleMovement()
        {
            if (this.cursors.left.isDown)
            {
                this.scale.x = -1; //flipped
                this.body.velocity.x = -200;
            }
            else if (this.cursors.right.isDown)
            {
                this.scale.x = 1; //facing default direction
                this.body.velocity.x = 200;
            }
            else if (this.cursors.up.isDown)
            {
                this.body.velocity.y = -200;
            }
            else if (this.cursors.down.isDown)
            {
                this.body.velocity.y = 200;
            }
        }

        private handleCollisions()
        {
            if (this.body.blocked.up || this.body.blocked.down || this.body.blocked.left || this.body.blocked.right) this.hitWall();
        }

        public hitWall = () =>
        {
            this.effect.play();

            //this.destroy(); // for removing object e.g. killing char
        }

        Animate()
        {
           this.animations.add("fly"); // whole sheet = fly animation
           this.animations.play("fly", this.animSpeed, true); // true -> loop forever
           this.animations.currentAnim.speed = this.animSpeed;
        }

    }

}
