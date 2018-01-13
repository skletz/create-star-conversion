var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
/// <reference path="../../resources/d_ts/phaser.d.ts"/>
var Sprite;
(function (Sprite) {
    var Character = (function (_super) {
        __extends(Character, _super);
        function Character(game, x, y) {
            var _this = _super.call(this, game, x, y, "BIRD_FLYING", 0) || this;
            _this.hitWall = function () {
                _this.effect.play();
                //this.destroy(); // for removing object e.g. killing char
            };
            _this.anchor.setTo(.5, 1); //so it flips around its middle
            _this.game = game;
            _this.animSpeed = 10; // 10 fps
            // arcade physics
            _this.game.physics.enable(_this, Phaser.Physics.ARCADE);
            _this.body.collideWorldBounds = true;
            _this.body.bounce.set(0.4);
            _this.body.setSize(74, 66, 12, 10);
            //this.body.velocity.setTo(0, -20);
            // sfx
            _this.effect = _this.game.add.audio('ding', 0.5, false);
            _this.Animate();
            _this.cursors = _this.game.input.keyboard.createCursorKeys();
            return _this;
            // usual way of adding to game (adding to group makes this obsolete)
            //this.game.add.existing(this);
        }
        Character.prototype.update = function () {
            this.handleMovement();
            this.handleCollisions();
        };
        Character.prototype.handleMovement = function () {
            if (this.cursors.left.isDown) {
                this.scale.x = -1; //flipped
                this.body.velocity.x = -200;
            }
            else if (this.cursors.right.isDown) {
                this.scale.x = 1; //facing default direction
                this.body.velocity.x = 200;
            }
            else if (this.cursors.up.isDown) {
                this.body.velocity.y = -200;
            }
            else if (this.cursors.down.isDown) {
                this.body.velocity.y = 200;
            }
        };
        Character.prototype.handleCollisions = function () {
            if (this.body.blocked.up || this.body.blocked.down || this.body.blocked.left || this.body.blocked.right)
                this.hitWall();
        };
        Character.prototype.Animate = function () {
            this.animations.add("fly"); // whole sheet = fly animation
            this.animations.play("fly", this.animSpeed, true); // true -> loop forever
            this.animations.currentAnim.speed = this.animSpeed;
        };
        Character.MAX_SPEED = 30; // 30 fps
        return Character;
    }(Phaser.Sprite));
    Sprite.Character = Character;
})(Sprite || (Sprite = {}));
//# sourceMappingURL=Character.js.map