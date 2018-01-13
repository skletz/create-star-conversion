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
    var Menu = (function (_super) {
        __extends(Menu, _super);
        function Menu() {
            return _super !== null && _super.apply(this, arguments) || this;
        }
        Menu.prototype.create = function () {
            this.bg = this.game.add.tileSprite(0, 0, this.game.cache.getImage('bg_menu').width, this.game.cache.getImage('bg_menu').height, 'bg_menu');
            this.bg.scale.x = Utils.getProportionalScale(this.game.width, this.game.cache.getImage('bg_menu').width);
            this.bg.scale.y = Utils.getProportionalScale(this.game.height, this.game.cache.getImage('bg_menu').height);
        };
        Menu.prototype.update = function () {
        };
        return Menu;
    }(Phaser.State));
    State.Menu = Menu;
})(State || (State = {}));
//# sourceMappingURL=Menu.js.map