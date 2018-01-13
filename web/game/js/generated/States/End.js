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
    var End = (function (_super) {
        __extends(End, _super);
        function End() {
            return _super !== null && _super.apply(this, arguments) || this;
        }
        End.prototype.create = function () {
            this.bg = this.game.add.tileSprite(0, 0, this.game.cache.getImage('bg_end').width, this.game.cache.getImage('bg_end').height, 'bg_end');
            this.bg.scale.x = Utils.getProportionalScale(this.game.width, this.game.cache.getImage('bg_end').width);
            this.bg.scale.y = Utils.getProportionalScale(this.game.height, this.game.cache.getImage('bg_end').height);
        };
        End.prototype.update = function () {
        };
        return End;
    }(Phaser.State));
    State.End = End;
})(State || (State = {}));
//# sourceMappingURL=End.js.map