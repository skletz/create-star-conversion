var Utils;
(function (Utils) {
    Utils.toInt = function (value) {
        return ~~value;
    };
    Utils.getProportionalScale = function (gameSize, assetSize) {
        if (assetSize >= gameSize)
            return gameSize / assetSize;
        else
            return assetSize / gameSize;
    };
})(Utils || (Utils = {}));
//# sourceMappingURL=utils.js.map