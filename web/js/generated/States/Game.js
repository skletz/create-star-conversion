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
/// <reference path="../Entities/Character.ts"/>
var State;
(function (State) {
    var Game = (function (_super) {
        __extends(Game, _super);
        function Game() {
            var _this = _super !== null && _super.apply(this, arguments) || this;
            _this.canvasZoom = 32;
            _this.isDown = false;
            _this.isErase = false;
            //  Dimensions
            _this.previewSize = 6;
            _this.spriteWidth = 8;
            _this.spriteHeight = 8;
            //  Palette
            _this.ci = 0;
            _this.color = 0;
            _this.palette = 0;
            _this.pmap = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F'];
            _this.colorindex = 6;
            _this.stopSounds = function () {
                console.log("Sounds stopped!");
                _this.sound.stopAll();
            };
            return _this;
        }
        Game.prototype.create = function () {
            // groups for sprite layering (prevents stacking by add order)
            this.bgGroup = this.game.add.group();
            this.frontGroup = this.game.add.group();
            // background
            this.bg = this.game.add.tileSprite(0, 0, this.game.cache.getImage('bg_game').width, this.game.cache.getImage('bg_game').height, 'bg_game');
            this.bg.scale.x = Utils.getProportionalScale(this.game.width, this.game.cache.getImage('bg_game').width);
            this.bg.scale.y = Utils.getProportionalScale(this.game.height, this.game.cache.getImage('bg_game').height);
            this.bgGroup.add(this.bg);
            // character
            // this.character = new Sprite.Character(this.game, this.game.width - this.game.width/2, this.game.height - this.game.height/2);
            // this.frontGroup.add(this.character);
            // start arcade physics
            //this.game.physics.startSystem(Phaser.Physics.ARCADE);
            //this.game.physics.arcade.gravity.y = 200;
            // sound
            // this.music  = this.game.add.audio('music', 1,true); // looping music
            // this.music.play();
            // stop all sounds on switching to different state    
            this.game.state.onStateChange.addOnce(this.stopSounds);
            // drawing
            this.createDrawingArea();
            this.game.input.mouse.capture = true;
            this.game.input.onDown.add(this.onDown, this);
            this.game.input.onUp.add(this.onUp, this);
            this.game.input.addMoveCallback(this.paint, this);
            this.resetData();
        };
        Game.prototype.onDown = function (pointer) {
            if (pointer.y <= 32) {
                // this.setColor(this.game.math.snapToFloor(pointer.x, 32) / 32);
            }
            else {
                this.isDown = true;
                if (pointer.rightButton.isDown) {
                    this.isErase = true;
                }
                else {
                    this.isErase = false;
                }
                this.paint(pointer);
            }
        };
        Game.prototype.onUp = function () {
            this.isDown = false;
        };
        Game.prototype.resetData = function () {
            this.data = [];
            for (var y = 0; y < this.spriteHeight; y++) {
                var a = [];
                for (var x = 0; x < this.spriteWidth; x++) {
                    a.push('.');
                }
                this.data.push(a);
            }
        };
        Game.prototype.refresh = function () {
            //  Update both the Canvas and Preview
            this.canvas.clear();
            // preview.clear();
            for (var y = 0; y < this.spriteHeight; y++) {
                for (var x = 0; x < this.spriteWidth; x++) {
                    var i = this.data[y][x];
                    if (i !== '.' && i !== ' ') {
                        this.color = this.game.create.palettes[this.palette][i];
                        this.canvas.rect(x * this.canvasZoom, y * this.canvasZoom, this.canvasZoom, this.canvasZoom, this.color);
                        // preview.rect(x * previewSize, y * previewSize, previewSize, previewSize, color);
                    }
                }
            }
        };
        Game.prototype.paint = function (pointer) {
            //  Get the grid loc from the pointer
            var math = this.game.math;
            var x = math.snapToFloor(pointer.x - this.canvasSprite.x, this.canvasZoom) / this.canvasZoom;
            var y = math.snapToFloor(pointer.y - this.canvasSprite.y, this.canvasZoom) / this.canvasZoom;
            if (x < 0 || x >= this.spriteWidth || y < 0 || y >= this.spriteHeight) {
                return;
            }
            //this.coords.text = "X: " + x + "\tY: " + y;
            console.log("X: " + x + "\tY: " + y);
            if (!this.isDown) {
                return;
            }
            if (this.isErase) {
                this.data[y][x] = '.';
                this.canvas.clear(x * this.canvasZoom, y * this.canvasZoom, this.canvasZoom, this.canvasZoom, this.color);
                //preview.clear(x * previewSize, y * previewSize, previewSize, previewSize, color);
            }
            else {
                this.data[y][x] = this.pmap[this.colorindex];
                this.canvas.rect(x * this.canvasZoom, y * this.canvasZoom, this.canvasZoom, this.canvasZoom, this.color);
                //preview.rect(x * previewSize, y * previewSize, previewSize, previewSize, this.color);
            }
        };
        Game.prototype.createDrawingArea = function () {
            this.game.create.grid('drawingGrid', 16 * this.canvasZoom, 16 * this.canvasZoom, this.canvasZoom, this.canvasZoom, 'rgba(0,191,243,0.8)');
            this.canvas = this.game.make.bitmapData(this.spriteWidth * this.canvasZoom, this.spriteHeight * this.canvasZoom);
            this.canvasBG = this.game.make.bitmapData(this.canvas.width + 2, this.canvas.height + 2);
            this.canvasBG.rect(0, 0, this.canvasBG.width, this.canvasBG.height, '#fff');
            this.canvasBG.rect(1, 1, this.canvasBG.width - 2, this.canvasBG.height - 2, '#3f5c67');
            var x = 10;
            var y = 64;
            this.canvasBG.addToWorld(x, y);
            this.canvasSprite = this.canvas.addToWorld(x + 1, y + 1);
            this.canvasGrid = this.game.add.sprite(x + 1, y + 1, 'drawingGrid');
            //this.canvasGrid.crop(new Phaser.Rectangle(0, 0, this.spriteWidth * canvasZoom, this.spriteHeight * canvasZoom));
        };
        Game.prototype.update = function () {
            this.refresh();
        };
        /* DEBUG (Show Bounding Box) */
        Game.prototype.render = function () {
            if (this.character) {
                this.game.debug.bodyInfo(this.character, 32, 32);
                var charact = this.frontGroup.getAt(0);
                if (charact != null)
                    this.game.debug.body(charact);
            }
        };
        return Game;
    }(Phaser.State));
    State.Game = Game;
})(State || (State = {}));
//# sourceMappingURL=Game.js.map