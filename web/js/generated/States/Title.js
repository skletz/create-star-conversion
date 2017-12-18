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
var State;
(function (State) {
    var Title = (function (_super) {
        __extends(Title, _super);
        function Title() {
            return _super !== null && _super.apply(this, arguments) || this;
        }
        Title.prototype.create = function () {
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
        };
        return Title;
    }(Phaser.State));
    State.Title = Title;
})(State || (State = {}));
//# sourceMappingURL=Title.js.map